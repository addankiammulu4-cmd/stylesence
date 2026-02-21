/* ── StyleAI Frontend Logic ────────────────────────────────────── */

// DOM refs
const navbar       = document.getElementById('navbar');
const hamburger    = document.getElementById('hamburger');
const navMenu      = document.querySelector('.nav-menu');
const fileInput    = document.getElementById('fileInput');
const uploadArea   = document.getElementById('uploadArea');
const browseBtn    = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loadingOverlay = document.getElementById('loadingOverlay');
const analyzeBtn   = document.getElementById('analyzeBtn');
const changePhotoBtn = document.getElementById('changePhotoBtn');
const uploadPanel  = document.getElementById('uploadPanel');
const resultsPanel = document.getElementById('resultsPanel');
const resetBtn     = document.getElementById('resetBtn');
const toast        = document.getElementById('toast');

let selectedFile = null;
let toastTimer   = null;

/* ── Navbar scroll ─────────────────────────────────────────────── */
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
  highlightNav();
});

function highlightNav() {
  const sections = document.querySelectorAll('section[id]');
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 120) current = s.id;
  });
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
  });
}

hamburger.addEventListener('click', () => {
  navMenu.classList.toggle('open');
});

/* ── Smooth close hamburger on link click ─────────────────────── */
document.querySelectorAll('.nav-link').forEach(l =>
  l.addEventListener('click', () => navMenu.classList.remove('open'))
);

/* ── File selection ────────────────────────────────────────────── */
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── Drag & Drop ───────────────────────────────────────────────── */
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  const allowed = ['image/png','image/jpeg','image/jpg','image/gif','image/webp'];
  if (!allowed.includes(file.type)) { showToast('Invalid file type. Please upload PNG, JPG, JPEG, GIF or WEBP.', 'error'); return; }
  if (file.size > 10 * 1024 * 1024) { showToast('File too large. Maximum size is 10 MB.', 'error'); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    uploadArea.style.display    = 'none';
    previewSection.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

changePhotoBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImage.src = '';
  uploadArea.style.display    = 'block';
  previewSection.style.display = 'none';
});

/* ── Analyse ───────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', analyzeStyle);

async function analyzeStyle() {
  if (!selectedFile) { showToast('Please select a photo first.', 'error'); return; }

  const gender = document.querySelector('input[name="gender"]:checked').value;

  // Show loading
  loadingOverlay.style.display = 'flex';
  analyzeBtn.disabled          = true;
  analyzeBtn.innerHTML         = '<i class="fas fa-spinner fa-spin"></i> Analysing…';

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('gender', gender);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Analysis failed');
    renderResults(data);

  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    loadingOverlay.style.display = 'none';
    analyzeBtn.disabled          = false;
    analyzeBtn.innerHTML         = '<i class="fas fa-magic"></i> Analyse My Style';
  }
}

/* ── Render Results ────────────────────────────────────────────── */
function renderResults(data) {
  const r = data.recommendations;

  // Skin tone
  document.getElementById('toneSwatch').style.background = data.hex_color;
  document.getElementById('toneTitle').textContent  = `${data.skin_tone} Skin Tone`;
  document.getElementById('toneDesc').textContent   = `Detected from your uploaded photo`;
  document.getElementById('toneRGB').textContent    = `RGB(${data.rgb.r}, ${data.rgb.g}, ${data.rgb.b}) · ${data.hex_color}`;

  // AI badge
  document.getElementById('aiBadge').style.display      = data.ai_powered ? 'flex' : 'none';
  document.getElementById('fallbackBadge').style.display = data.ai_powered ? 'none' : 'flex';

  // Dress codes
  const dc = document.getElementById('dressCodes');
  dc.innerHTML = (r.dress_codes || [])
    .map(d => `<span class="dress-code-tag">${d}</span>`).join('');

  // Suggested outfit
  document.getElementById('suggestedOutfit').textContent = r.suggested_outfit || '';

  // Shirt
  document.getElementById('shirtDetails').innerHTML = buildDetailRows(r.shirt);
  // Bottom
  document.getElementById('bottomDetails').innerHTML = buildDetailRows(r.bottom);
  // Shoes
  document.getElementById('shoesDetails').innerHTML = buildDetailRows(r.shoes);

  // Hairstyle
  document.getElementById('hairstyleStyle').textContent  = r.hairstyle?.style || '';
  document.getElementById('hairstyleHowto').textContent  = r.hairstyle?.howto || '';

  // Accessories
  const ag = document.getElementById('accessoriesGrid');
  ag.innerHTML = (r.accessories || [])
    .map(a => `<span class="accessory-tag"><i class="fas fa-circle-dot" style="font-size:.5rem;color:var(--accent)"></i>${a}</span>`).join('');

  // Colour palette
  const pd = document.getElementById('paletteDisplay');
  const palette = r.color_palette || {};
  pd.innerHTML = Object.entries(palette).map(([role, name]) => `
    <div class="palette-item">
      <div class="palette-swatch" style="background:${colorNameToHex(name)}"></div>
      <div class="palette-label">${role}</div>
      <div class="palette-name">${name}</div>
    </div>`).join('');

  // Why it works
  document.getElementById('whyItWorks').textContent = r.why_it_works || '';

  // Shopping
  const pg = document.getElementById('productsGrid');
  pg.innerHTML = (data.products || []).map(p => `
    <div class="product-card">
      <div class="product-store">${p.store}</div>
      <div class="product-name">${p.name}</div>
      <div class="product-price">${p.price}</div>
      <a href="${p.url}" target="_blank" rel="noopener" class="product-btn">
        <i class="fas fa-external-link-alt"></i> Shop Now
      </a>
    </div>`).join('');

  // Show results
  uploadPanel.style.display  = 'none';
  resultsPanel.style.display = 'flex';

  // Smooth scroll
  document.getElementById('upload').scrollIntoView({ behavior: 'smooth', block: 'start' });
  showToast(`${data.skin_tone} skin tone detected — your style profile is ready! ✦`, 'success');
}

function buildDetailRows(obj) {
  if (!obj) return '';
  return Object.entries(obj).map(([k, v]) =>
    `<div class="outfit-detail-row"><strong>${cap(k)}:</strong> ${v}</div>`
  ).join('');
}

function cap(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/* ── Colour name → hex approximation ─────────────────────────── */
const COLOR_MAP = {
  'navy blue':'#1a2a5e', 'navy':'#1a2a5e', 'light grey':'#b0b3b8',
  'grey':'#808080', 'gray':'#808080', 'burgundy':'#800020',
  'olive green':'#6b6b30', 'olive':'#6b6b30', 'cream':'#fffdd0',
  'rust':'#b7410e', 'white':'#f5f5f0', 'khaki':'#c3b091',
  'teal':'#008080', 'off-white':'#faf9f6', 'gold':'#ffd700',
  'royal blue':'#4169e1', 'orange':'#ff8c00', 'black':'#1a1a1a',
  'dusty rose':'#dcb4b4', 'ivory':'#fffff0', 'blush pink':'#ffb6c1',
  'terracotta':'#e2725b', 'sage green':'#8fae88', 'sage':'#8fae88',
  'bronze':'#cd7f32', 'emerald':'#50c878', 'champagne':'#f7e7ce',
  'copper':'#b87333', 'fuchsia':'#ff00ff', 'cobalt':'#0047ab',
  'tan':'#d2b48c', 'charcoal':'#36454f', 'nude':'#e3bc9a',
  'dark brown':'#4a2c2a', 'light blue':'#add8e6',
  'royal blue fitted':'#4169e1', 'emerald green':'#50c878',
};
function colorNameToHex(name) {
  const key = (name || '').toLowerCase().trim();
  return COLOR_MAP[key] || COLOR_MAP[key.split(' ').slice(0,2).join(' ')] || '#c9a96e';
}

/* ── Reset ─────────────────────────────────────────────────────── */
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value   = '';
  previewImage.src  = '';
  previewSection.style.display = 'none';
  uploadArea.style.display    = 'block';
  uploadPanel.style.display   = 'block';
  resultsPanel.style.display  = 'none';
  document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
});

/* ── Toast ─────────────────────────────────────────────────────── */
function showToast(msg, type = 'info') {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className   = `toast ${type} show`;
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4500);
}

/* ── Entrance animations ───────────────────────────────────────── */
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity   = '1';
      e.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card, .about-card, .step').forEach(el => {
  el.style.opacity   = '0';
  el.style.transform = 'translateY(24px)';
  el.style.transition = 'opacity .6s ease, transform .6s ease';
  observer.observe(el);
});
