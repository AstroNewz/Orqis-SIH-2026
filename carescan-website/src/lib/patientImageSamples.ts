/**
 * Clinical sample SVG illustrations for patient anomaly screening preview
 */

// 1. Normal Healthy Pink Mucosa
export const SAMPLE_IMAGE_HEALTHY = `data:image/svg+xml;utf8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
  <defs>
    <radialGradient id="healthyGrad" cx="50%" cy="50%" r="70%">
      <stop offset="0%" stop-color="#fda4af" />
      <stop offset="60%" stop-color="#f43f5e" />
      <stop offset="100%" stop-color="#be123c" />
    </radialGradient>
    <radialGradient id="mucosaGlow" cx="45%" cy="40%" r="40%">
      <stop offset="0%" stop-color="#fecdd3" stop-opacity="0.8" />
      <stop offset="100%" stop-color="#f43f5e" stop-opacity="0" />
    </radialGradient>
    <filter id="noise">
      <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="3" result="noise" />
      <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.15 0"/>
      <feComposite in2="SourceGraphic" in="gl" operator="in" />
    </filter>
  </defs>
  <!-- Base mucosal tissue -->
  <rect width="400" height="300" fill="url(#healthyGrad)" />
  <rect width="400" height="300" fill="url(#mucosaGlow)" />
  
  <!-- Subtle vascular capillaries -->
  <path d="M 40,150 Q 120,130 180,160 T 320,140" fill="none" stroke="#9f1239" stroke-width="1.2" opacity="0.4" />
  <path d="M 100,80 Q 160,110 240,90 T 360,120" fill="none" stroke="#881337" stroke-width="0.8" opacity="0.3" />
  <path d="M 80,220 Q 180,200 260,230 T 350,200" fill="none" stroke="#9f1239" stroke-width="1" opacity="0.35" />
  
  <!-- Healthy Tissue Overlay Annotation -->
  <rect x="20" y="20" width="160" height="28" rx="8" fill="#0f172a" fill-opacity="0.75" />
  <text x="30" y="38" fill="#34d399" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">● Healthy Mucosa (Benign)</text>
  
  <!-- Smooth Grid Crosshairs -->
  <circle cx="200" cy="150" r="45" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6" />
  <line x1="200" y1="95" x2="200" y2="205" stroke="#34d399" stroke-width="1" opacity="0.4" />
  <line x1="145" y1="150" x2="255" y2="150" stroke="#34d399" stroke-width="1" opacity="0.4" />
</svg>
`)}`;

// 2. Lichenoid Reticular Striae (Moderate Risk / Reactive)
export const SAMPLE_IMAGE_LICHENOID = `data:image/svg+xml;utf8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
  <defs>
    <radialGradient id="modGrad" cx="50%" cy="50%" r="70%">
      <stop offset="0%" stop-color="#fb7185" />
      <stop offset="70%" stop-color="#e11d48" />
      <stop offset="100%" stop-color="#9f1239" />
    </radialGradient>
    <radialGradient id="erythema" cx="50%" cy="50%" r="35%">
      <stop offset="0%" stop-color="#e11d48" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#fb7185" stop-opacity="0" />
    </radialGradient>
  </defs>
  <!-- Base tissue -->
  <rect width="400" height="300" fill="url(#modGrad)" />
  <circle cx="200" cy="150" r="70" fill="url(#erythema)" />
  
  <!-- Wickham's Striae (Reticular white lace pattern) -->
  <path d="M 160,130 C 180,120 190,140 210,130 S 240,145 250,135" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" opacity="0.85" />
  <path d="M 150,150 C 170,165 195,145 220,160 S 245,150 255,165" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" opacity="0.8" />
  <path d="M 175,120 C 185,140 170,160 180,180" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.75" />
  <path d="M 225,125 C 215,145 230,165 220,185" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" opacity="0.75" />
  <circle cx="195" cy="148" r="4" fill="#ffffff" opacity="0.9" />
  <circle cx="215" cy="155" r="3" fill="#ffffff" opacity="0.85" />
  
  <!-- Annotation Badge -->
  <rect x="20" y="20" width="180" height="28" rx="8" fill="#0f172a" fill-opacity="0.75" />
  <text x="30" y="38" fill="#fbbf24" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">● Reticular Striae (Moderate)</text>
  
  <!-- Crosshairs -->
  <circle cx="200" cy="150" r="55" fill="none" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.7" />
</svg>
`)}`;

// 3. Homogeneous Leukoplakia / Keratotic Plaque (High Risk Anomaly)
export const SAMPLE_IMAGE_LEUKOPLAKIA = `data:image/svg+xml;utf8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
  <defs>
    <radialGradient id="highGrad" cx="50%" cy="50%" r="70%">
      <stop offset="0%" stop-color="#e11d48" />
      <stop offset="60%" stop-color="#9f1239" />
      <stop offset="100%" stop-color="#4c0519" />
    </radialGradient>
    <radialGradient id="plaqueGrad" cx="45%" cy="45%" r="60%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="70%" stop-color="#f1f5f9" />
      <stop offset="100%" stop-color="#cbd5e1" />
    </radialGradient>
  </defs>
  <!-- Base tissue -->
  <rect width="400" height="300" fill="url(#highGrad)" />
  
  <!-- Elevated Keratotic Plaque with Irregular Borders -->
  <path d="M 140,120 C 160,95 210,90 250,110 C 280,125 275,170 255,190 C 230,210 170,215 145,185 C 125,160 120,135 140,120 Z" 
        fill="url(#plaqueGrad)" stroke="#fda4af" stroke-width="3" filter="drop-shadow(0 4px 6px rgba(0,0,0,0.3))" />
        
  <!-- Central erythematous micro-fissure / ulceration -->
  <path d="M 185,145 Q 200,135 215,150 T 230,165" fill="none" stroke="#be123c" stroke-width="2.5" stroke-linecap="round" />
  <circle cx="205" cy="150" r="6" fill="#be123c" opacity="0.8" />
  
  <!-- Surrounding inflammation halo -->
  <path d="M 130,110 C 155,80 220,75 265,100 C 300,120 295,185 270,205 C 240,230 160,235 130,200 C 110,170 105,130 130,110 Z" 
        fill="none" stroke="#f43f5e" stroke-width="2" stroke-dasharray="6 4" opacity="0.75" />
        
  <!-- Annotation Badge -->
  <rect x="20" y="20" width="190" height="28" rx="8" fill="#0f172a" fill-opacity="0.75" />
  <text x="30" y="38" fill="#f43f5e" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">● Keratotic Plaque (High Risk)</text>
  
  <!-- High Risk Target Crosshair -->
  <circle cx="200" cy="150" r="65" fill="none" stroke="#f43f5e" stroke-width="2" stroke-dasharray="5 3" opacity="0.8" />
  <line x1="200" y1="80" x2="200" y2="220" stroke="#f43f5e" stroke-width="1.2" opacity="0.6" />
  <line x1="130" y1="150" x2="270" y2="150" stroke="#f43f5e" stroke-width="1.2" opacity="0.6" />
</svg>
`)}`;
