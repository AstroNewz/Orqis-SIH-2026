import React from 'react';

export const QuantumOrb: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`relative flex items-center justify-center select-none ${className}`}>
      {/* Background Soft Glow */}
      <div className="absolute w-72 h-72 sm:w-96 sm:h-96 rounded-full bg-gradient-to-tr from-indigo-200/50 via-teal-100/40 to-purple-200/50 blur-3xl -z-10 animate-pulse-glow" />

      {/* Main SVG Composition */}
      <svg
        viewBox="0 0 500 500"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full max-w-[420px] sm:max-w-[480px] h-auto drop-shadow-xl"
      >
        <defs>
          <linearGradient id="orbGradient" x1="0" y1="0" x2="500" y2="500" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#4F46E5" stopOpacity="0.85" />
            <stop offset="50%" stopColor="#0D9488" stopOpacity="0.75" />
            <stop offset="100%" stopColor="#7C3AED" stopOpacity="0.9" />
          </linearGradient>

          <linearGradient id="meshGrad" x1="150" y1="150" x2="350" y2="350" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#818CF8" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#2DD4BF" stopOpacity="0.4" />
          </linearGradient>

          <radialGradient id="centerPulse" cx="250" cy="250" r="150" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#EEF2FF" stopOpacity="0.9" />
            <stop offset="45%" stopColor="#C7D2FE" stopOpacity="0.4" />
            <stop offset="85%" stopColor="#4F46E5" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#4F46E5" stopOpacity="0" />
          </radialGradient>

          <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="100" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#2DD4BF" stopOpacity="0" />
            <stop offset="50%" stopColor="#2DD4BF" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#2DD4BF" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Outer Orbit 1 */}
        <circle
          cx="250"
          cy="250"
          r="210"
          stroke="#E2E8F0"
          strokeWidth="1.5"
          strokeDasharray="4 8"
          className="opacity-70"
        />

        {/* Outer Orbit 2 - Tilted Ellipse */}
        <ellipse
          cx="250"
          cy="250"
          rx="220"
          ry="110"
          stroke="url(#orbGradient)"
          strokeWidth="1.5"
          strokeDasharray="6 6"
          transform="rotate(35 250 250)"
          className="opacity-60"
        />

        {/* Orbit 3 - Reverse Tilted Ellipse */}
        <ellipse
          cx="250"
          cy="250"
          rx="220"
          ry="110"
          stroke="url(#orbGradient)"
          strokeWidth="1.5"
          strokeDasharray="6 6"
          transform="rotate(-35 250 250)"
          className="opacity-60"
        />

        {/* Central Organic Biological Cellular Core */}
        <circle cx="250" cy="250" r="140" fill="url(#centerPulse)" />
        <circle cx="250" cy="250" r="140" stroke="#C7D2FE" strokeWidth="1.5" className="opacity-60" />

        {/* Inner Cellular Node Mesh (Hexagonal & Biological Lattice) */}
        <g stroke="url(#meshGrad)" strokeWidth="1.5">
          <line x1="250" y1="170" x2="310" y2="210" />
          <line x1="310" y1="210" x2="310" y2="290" />
          <line x1="310" y1="290" x2="250" y2="330" />
          <line x1="250" y1="330" x2="190" y2="290" />
          <line x1="190" y1="290" x2="190" y2="210" />
          <line x1="190" y1="210" x2="250" y2="170" />

          {/* Cross connecting diagonals */}
          <line x1="250" y1="170" x2="250" y2="330" strokeDasharray="2 4" />
          <line x1="190" y1="250" x2="310" y2="250" strokeDasharray="2 4" />
          <line x1="190" y1="210" x2="310" y2="290" strokeDasharray="3 3" />
          <line x1="190" y1="290" x2="310" y2="210" strokeDasharray="3 3" />
        </g>

        {/* Mesh Nodes with Glow */}
        {[
          { cx: 250, cy: 170, color: '#4F46E5', label: 'q₀' },
          { cx: 310, cy: 210, color: '#0D9488', label: 'q₁' },
          { cx: 310, cy: 290, color: '#7C3AED', label: 'q₂' },
          { cx: 250, cy: 330, color: '#4F46E5', label: 'q₃' },
          { cx: 190, cy: 290, color: '#0D9488', label: 'q₄' },
          { cx: 190, cy: 210, color: '#7C3AED', label: 'q₅' },
          { cx: 250, cy: 250, color: '#4F46E5', label: 'ψ' },
        ].map((node, i) => (
          <g key={i}>
            <circle cx={node.cx} cy={node.cy} r="8" fill="#FFFFFF" stroke={node.color} strokeWidth="2.5" />
            <circle cx={node.cx} cy={node.cy} r="3" fill={node.color} />
          </g>
        ))}

        {/* Orbiting Quantum Electrons / Qubit Phase Markers */}
        <g transform="rotate(35 250 250)">
          <circle cx="470" cy="250" r="6" fill="#4F46E5" />
          <circle cx="30" cy="250" r="6" fill="#0D9488" />
        </g>

        <g transform="rotate(-35 250 250)">
          <circle cx="470" cy="250" r="6" fill="#7C3AED" />
          <circle cx="30" cy="250" r="6" fill="#4F46E5" />
        </g>

        {/* Dynamic Scanning Reticle Overlay */}
        <g stroke="#0D9488" strokeWidth="1.5" opacity="0.8">
          {/* Top-left corner */}
          <path d="M 170 160 L 150 160 L 150 180" />
          {/* Top-right corner */}
          <path d="M 330 160 L 350 160 L 350 180" />
          {/* Bottom-left corner */}
          <path d="M 170 340 L 150 340 L 150 320" />
          {/* Bottom-right corner */}
          <path d="M 330 340 L 350 340 L 350 320" />
        </g>

        {/* Center Target Indicator */}
        <circle cx="250" cy="250" r="28" stroke="#0D9488" strokeWidth="1.5" strokeDasharray="3 3" />
        <circle cx="250" cy="250" r="5" fill="#0D9488" />
      </svg>

      {/* Floating Telemetry Badge 1 - Left */}
      <div className="absolute -left-2 sm:left-4 top-1/4 bg-white/95 dark:bg-slate-900/95 px-3.5 py-2 rounded-2xl shadow-lg flex items-center gap-2.5 border border-slate-200/90 dark:border-slate-700/90 animate-bounce [animation-duration:6s] backdrop-blur-md transition-colors">
        <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping shrink-0" />
        <div className="text-left">
          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-300">8-Qubit State</p>
          <p className="text-xs font-bold text-slate-900 dark:text-teal-300 font-mono">Ry(θ), θ ≤ π/2</p>
        </div>
      </div>

      {/* Floating Telemetry Badge 2 - Right */}
      <div className="absolute -right-2 sm:right-4 bottom-1/4 bg-white/95 dark:bg-slate-900/95 px-3.5 py-2 rounded-2xl shadow-lg flex items-center gap-2.5 border border-slate-200/90 dark:border-slate-700/90 animate-bounce [animation-duration:7s] backdrop-blur-md transition-colors">
        <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 shrink-0" />
        <div className="text-left">
          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-300">Z Observables</p>
          <p className="text-xs font-bold text-slate-900 dark:text-indigo-300 font-mono">8 ⟨Z⟩ + 8 ⟨ZZ⟩</p>
        </div>
      </div>
    </div>
  );
};
