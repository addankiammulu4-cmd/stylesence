import os
import base64
import json
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Config ──────────────────────────────────────────────────────────────────
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB
UPLOAD_FOLDER   = os.path.join('static', 'uploads')
ALLOWED_EXT     = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
GROQ_API_KEY    = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL      = 'llama-3.3-70b-versatile'
MAX_TOKENS      = 1200
TEMPERATURE     = 0.7

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs(os.path.join('static', 'outputs'), exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def detect_skin_tone(image_path):
    """
    Detect dominant skin tone from a facial image using OpenCV + HSV colour
    analysis.  Returns (category, hex_color, rgb_tuple).
    """
    try:
        import cv2
        from PIL import Image

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")

        # ── Try face-crop first ───────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            # centre-crop to reduce hair / background
            cx, cy = x + w // 4, y + h // 4
            cw, ch = w // 2, h // 2
            roi = img[cy:cy+ch, cx:cx+cw]
        else:
            # Fall back to centre-quarter of the image
            hh, ww = img.shape[:2]
            roi = img[hh//4:3*hh//4, ww//4:3*ww//4]

        # ── HSV skin-pixel mask ──────────────────────────────────────────
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70],  dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
        skin  = cv2.bitwise_and(roi, roi, mask=mask)

        # Mean over masked pixels only
        pixels = skin[mask > 0]
        if len(pixels) < 100:
            # Fall back to mean of full ROI
            mean_bgr = roi.mean(axis=(0, 1))
        else:
            mean_bgr = pixels.reshape(-1, 3).mean(axis=0)

        b, g, r = [int(c) for c in mean_bgr]
        hex_color = f'#{r:02x}{g:02x}{b:02x}'

        # ── Classify by perceived luminance ─────────────────────────────
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance > 200:
            category = 'Fair'
        elif luminance > 160:
            category = 'Medium'
        elif luminance > 110:
            category = 'Olive'
        else:
            category = 'Deep'

        return category, hex_color, (r, g, b)

    except Exception as e:
        print(f"Skin tone detection error: {e}")
        return 'Medium', '#c68642', (198, 134, 66)


def get_groq_recommendations(skin_tone, gender, rgb):
    """Call Groq API and return a structured JSON recommendation."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""You are an expert personal fashion stylist. A user has the following profile:
- Skin Tone: {skin_tone}
- RGB values: R={rgb[0]}, G={rgb[1]}, B={rgb[2]}
- Gender: {gender}

Provide comprehensive, personalized fashion recommendations in STRICT JSON format only.
No extra text outside the JSON.

Return EXACTLY this structure:
{{
  "dress_codes": ["Formal", "Business Casual", "Smart Casual", "Party"],
  "suggested_outfit": "A detailed 2-sentence outfit description",
  "shirt": {{
    "color": "color name",
    "type": "shirt type",
    "brand": "brand suggestion",
    "fabric": "fabric type"
  }},
  "bottom": {{
    "color": "color name",
    "type": "pants/skirt type",
    "brand": "brand suggestion",
    "fabric": "fabric type"
  }},
  "shoes": {{
    "color": "color name",
    "type": "shoe type",
    "brand": "brand suggestion"
  }},
  "hairstyle": {{
    "style": "hairstyle name",
    "howto": "2-sentence how-to and maintenance tip"
  }},
  "accessories": ["accessory 1", "accessory 2", "accessory 3", "accessory 4"],
  "color_palette": {{
    "primary": "color name",
    "secondary": "color name",
    "accent": "color name"
  }},
  "why_it_works": "2-3 sentence explanation of why these recommendations suit this skin tone"
}}"""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as e:
        print(f"Groq API error: {e}")
        return get_fallback_recommendations(skin_tone, gender)


def get_fallback_recommendations(skin_tone, gender):
    """Return sensible defaults when the API call fails."""
    male_data = {
        "Fair":   {"primary":"Navy Blue","secondary":"Light Grey","accent":"Burgundy",
                   "shirt_color":"Light Blue","shirt_type":"Oxford Button-Down","bottom_color":"Charcoal","bottom_type":"Slim Chinos"},
        "Medium": {"primary":"Olive Green","secondary":"Cream","accent":"Rust",
                   "shirt_color":"White","shirt_type":"Linen Shirt","bottom_color":"Khaki","bottom_type":"Straight Trousers"},
        "Olive":  {"primary":"Teal","secondary":"Off-White","accent":"Gold",
                   "shirt_color":"Teal","shirt_type":"Polo Shirt","bottom_color":"Dark Brown","bottom_type":"Chinos"},
        "Deep":   {"primary":"Royal Blue","secondary":"White","accent":"Orange",
                   "shirt_color":"Royal Blue","shirt_type":"Fitted T-Shirt","bottom_color":"Black","bottom_type":"Slim Jeans"},
    }
    female_data = {
        "Fair":   {"primary":"Dusty Rose","secondary":"Ivory","accent":"Gold",
                   "shirt_color":"Blush Pink","shirt_type":"Flowy Blouse","bottom_color":"Cream","bottom_type":"Wide-Leg Trousers"},
        "Medium": {"primary":"Terracotta","secondary":"Sage Green","accent":"Bronze",
                   "shirt_color":"Terracotta","shirt_type":"Wrap Top","bottom_color":"Sage","bottom_type":"A-Line Skirt"},
        "Olive":  {"primary":"Emerald","secondary":"Champagne","accent":"Copper",
                   "shirt_color":"Emerald","shirt_type":"Fitted Blouse","bottom_color":"Nude","bottom_type":"Pencil Skirt"},
        "Deep":   {"primary":"Fuchsia","secondary":"Cobalt","accent":"Gold",
                   "shirt_color":"Fuchsia","shirt_type":"Wrap Dress top","bottom_color":"Cobalt","bottom_type":"Maxi Skirt"},
    }

    d = male_data[skin_tone] if gender == "Male" else female_data[skin_tone]
    accessories = (["Leather Watch","Silver Chain","Brown Belt","Aviator Sunglasses"]
                   if gender == "Male"
                   else ["Gold Hoop Earrings","Layered Necklace","Leather Handbag","Silk Scarf"])

    return {
        "dress_codes": ["Formal","Business Casual","Smart Casual","Party"],
        "suggested_outfit": f"A {d['shirt_color']} {d['shirt_type']} paired with {d['bottom_color']} {d['bottom_type']} creates a balanced, elegant look suited to your {skin_tone} skin tone.",
        "shirt":  {"color":d['shirt_color'],"type":d['shirt_type'],"brand":"Calvin Klein","fabric":"Cotton"},
        "bottom": {"color":d['bottom_color'],"type":d['bottom_type'],"brand":"Zara","fabric":"Cotton Blend"},
        "shoes":  {"color":"Tan","type":"Loafers" if gender=="Male" else "Block Heels","brand":"Clarks"},
        "hairstyle": {"style":"Classic Side Part" if gender=="Male" else "Soft Waves",
                      "howto":"Apply light pomade and comb to the side for a polished finish. Trim every 3-4 weeks to maintain shape." if gender=="Male"
                              else "Use a large-barrel curling wand for loose waves. Finish with shine spray and trim ends monthly."},
        "accessories": accessories,
        "color_palette": {"primary":d['primary'],"secondary":d['secondary'],"accent":d['accent']},
        "why_it_works": f"These colours create a harmonious contrast with your {skin_tone} skin tone, enhancing your natural complexion. The fabric choices add texture while keeping the look refined and contemporary.",
    }


def get_shopping_links(skin_tone, gender):
    """Return curated product dicts with Indian e-commerce links."""
    products = {
        "Male": {
            "Fair":  [
                {"name":"Light Blue Oxford Shirt","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=light+blue+oxford+shirt+men","price":"₹1,299"},
                {"name":"Charcoal Slim Chinos","store":"Myntra",
                 "url":"https://www.myntra.com/chinos/men","price":"₹1,799"},
                {"name":"Tan Derby Shoes","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=tan+derby+shoes+men","price":"₹2,499"},
                {"name":"Navy Blazer","store":"Zara",
                 "url":"https://www.zara.com/in/en/man-blazers-l767.html","price":"₹5,990"},
                {"name":"Leather Belt","store":"Myntra",
                 "url":"https://www.myntra.com/belts/men","price":"₹799"},
            ],
            "Medium":[
                {"name":"White Linen Shirt","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=white+linen+shirt+men","price":"₹999"},
                {"name":"Khaki Straight Trousers","store":"Myntra",
                 "url":"https://www.myntra.com/trousers/men","price":"₹1,599"},
                {"name":"White Sneakers","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=white+sneakers+men","price":"₹1,999"},
                {"name":"Olive Jacket","store":"Zara",
                 "url":"https://www.zara.com/in/en/man-outerwear-l775.html","price":"₹4,990"},
                {"name":"Canvas Watch","store":"Myntra",
                 "url":"https://www.myntra.com/watches/men","price":"₹2,299"},
            ],
            "Olive": [
                {"name":"Teal Polo Shirt","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=teal+polo+shirt+men","price":"₹899"},
                {"name":"Dark Brown Chinos","store":"Myntra",
                 "url":"https://www.myntra.com/chinos/men","price":"₹1,699"},
                {"name":"Brown Loafers","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=brown+loafers+men","price":"₹2,299"},
                {"name":"Cream Linen Blazer","store":"Zara",
                 "url":"https://www.zara.com/in/en/man-blazers-l767.html","price":"₹5,490"},
                {"name":"Gold Watch","store":"Myntra",
                 "url":"https://www.myntra.com/watches/men","price":"₹3,499"},
            ],
            "Deep":  [
                {"name":"Royal Blue Fitted Shirt","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=royal+blue+shirt+men","price":"₹1,099"},
                {"name":"Black Slim Jeans","store":"Myntra",
                 "url":"https://www.myntra.com/jeans/men","price":"₹1,499"},
                {"name":"White Leather Sneakers","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=white+sneakers+men","price":"₹2,199"},
                {"name":"Orange Accent Tee","store":"Zara",
                 "url":"https://www.zara.com/in/en/man-tshirts-l855.html","price":"₹1,490"},
                {"name":"Silver Necklace","store":"Myntra",
                 "url":"https://www.myntra.com/necklaces/men","price":"₹999"},
            ],
        },
        "Female":{
            "Fair":  [
                {"name":"Blush Pink Flowy Blouse","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=blush+pink+blouse+women","price":"₹999"},
                {"name":"Cream Wide-Leg Trousers","store":"Myntra",
                 "url":"https://www.myntra.com/trousers/women","price":"₹1,599"},
                {"name":"Nude Block Heels","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=nude+block+heels+women","price":"₹1,799"},
                {"name":"Dusty Rose Kurti","store":"Myntra",
                 "url":"https://www.myntra.com/kurtis/women","price":"₹1,299"},
                {"name":"Gold Hoop Earrings","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=gold+hoop+earrings+women","price":"₹599"},
            ],
            "Medium":[
                {"name":"Terracotta Wrap Top","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=terracotta+top+women","price":"₹899"},
                {"name":"Sage Green A-Line Skirt","store":"Myntra",
                 "url":"https://www.myntra.com/skirts/women","price":"₹1,399"},
                {"name":"Bronze Sandals","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=bronze+sandals+women","price":"₹1,499"},
                {"name":"Ethnic Printed Kurta","store":"Myntra",
                 "url":"https://www.myntra.com/kurtas/women","price":"₹1,199"},
                {"name":"Layered Necklace","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=layered+necklace+women","price":"₹799"},
            ],
            "Olive": [
                {"name":"Emerald Fitted Blouse","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=emerald+green+blouse+women","price":"₹1,099"},
                {"name":"Nude Pencil Skirt","store":"Myntra",
                 "url":"https://www.myntra.com/skirts/women","price":"₹1,299"},
                {"name":"Copper Wedge Heels","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=wedge+heels+women","price":"₹1,899"},
                {"name":"Champagne Evening Dress","store":"Zara",
                 "url":"https://www.zara.com/in/en/woman-dresses-l1066.html","price":"₹3,990"},
                {"name":"Copper Bracelet Set","store":"Myntra",
                 "url":"https://www.myntra.com/bracelets/women","price":"₹699"},
            ],
            "Deep":  [
                {"name":"Fuchsia Wrap Dress","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=fuchsia+dress+women","price":"₹1,499"},
                {"name":"Cobalt Maxi Skirt","store":"Myntra",
                 "url":"https://www.myntra.com/skirts/women","price":"₹1,699"},
                {"name":"Gold Strappy Heels","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=gold+heels+women","price":"₹2,199"},
                {"name":"Bold Print Kurta Set","store":"Myntra",
                 "url":"https://www.myntra.com/kurta-sets/women","price":"₹2,499"},
                {"name":"Gold Statement Earrings","store":"Amazon.in",
                 "url":"https://www.amazon.in/s?k=gold+statement+earrings+women","price":"₹899"},
            ],
        },
    }
    return products.get(gender, products["Male"]).get(skin_tone, [])


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    groq_configured = bool(GROQ_API_KEY and GROQ_API_KEY != 'your_groq_api_key_here')
    return jsonify({'status': 'running', 'groq_configured': groq_configured})


@app.route('/analyze', methods=['POST'])
def analyze():
    # ── Validate upload ──────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file   = request.files['file']
    gender = request.form.get('gender', 'Male')

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed. Use PNG, JPG, JPEG, GIF or WEBP'}), 400

    # ── Save file ────────────────────────────────────────────────────────
    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # ── Skin tone detection ──────────────────────────────────────────
        skin_tone, hex_color, rgb = detect_skin_tone(filepath)

        # ── AI recommendations ───────────────────────────────────────────
        groq_ok = bool(GROQ_API_KEY and GROQ_API_KEY != 'your_groq_api_key_here')
        if groq_ok:
            recommendations = get_groq_recommendations(skin_tone, gender, rgb)
        else:
            recommendations = get_fallback_recommendations(skin_tone, gender)

        # ── Shopping links ───────────────────────────────────────────────
        products = get_shopping_links(skin_tone, gender)

        return jsonify({
            'success':         True,
            'skin_tone':       skin_tone,
            'hex_color':       hex_color,
            'rgb':             {'r': rgb[0], 'g': rgb[1], 'b': rgb[2]},
            'gender':          gender,
            'recommendations': recommendations,
            'products':        products,
            'image_url':       f'/static/uploads/{filename}',
            'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ai_powered':      groq_ok,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    groq_ok = bool(GROQ_API_KEY and GROQ_API_KEY != 'your_groq_api_key_here')
    print("\n" + "="*60)
    print("  👗  STYLE AI — Personal Fashion Styling Advisor")
    print("="*60)
    print(f"  Groq API : {'✅ Configured' if groq_ok else '⚠️  Not set (using fallback)'}")
    print(f"  Server   : http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
