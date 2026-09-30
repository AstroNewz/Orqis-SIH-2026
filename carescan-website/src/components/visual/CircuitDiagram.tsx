'use client';

import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Info, Sparkles } from 'lucide-react';

interface GateInfo {
  name: string;
  type: string;
  math: string;
  description: string;
  clinicalImpact: string;
}

const GATE_DETAILS: Record<string, GateInfo> = {
  prep: {
    name: '1. PCA-8 Reduction → Encoding Angles',
    type: 'TRAIN-Fitted Preprocessing',
    math: 'θⱼ = (π/2) · s(zⱼ),  z = PCA₈(ROI descriptor)',
    description:
      'The frozen localizer\'s ROI descriptor is reduced to 8 dimensions by a PCA fitted on the TRAIN partition only, then each component is scaled into one rotation angle. The ceiling π/2 was selected on nested patient-grouped cross-validation over TRAIN — never against validation or test.',
    clinicalImpact:
      'Fixes the encoding before any label is seen, so the quantum stage cannot have been tuned toward a score.',
  },
  rot: {
    name: '2. Ry Angle Encoding, Re-Uploaded Twice',
    type: 'Fixed Rotations — 0 Trainable Gates',
    math: '|ψ(x)⟩ = ⨂ⱼ R_y(θⱼ)|0⟩,  B = 2 blocks',
    description:
      'Each angle drives a single Ry rotation, re-uploaded across two blocks. Nothing here learns: there are zero trainable quantum parameters. Two blocks are required for the ring to entangle at all, and they fold the response like cos(2θ) — which is exactly why the angle ceiling is π/2 and not π.',
    clinicalImpact:
      'A fixed lens rather than an adjustable one — the circuit cannot be bent until the picture flatters the result.',
  },
  cnot: {
    name: '3. Nearest-Neighbour CZ Ring',
    type: 'Entanglement, Measured Not Assumed',
    math: 'Cⱼₖ = ⟨ZⱼZₖ⟩ − ⟨Zⱼ⟩⟨Zₖ⟩,  mean |C| = 0.184',
    description:
      'A CZ ring between neighbouring qubits, sandwiched between the two encoding blocks. Whether it does anything is measured, not asserted: at one block the ring commutes through every diagonal Z-string and contributes identically zero, which would disqualify the circuit as quantum content.',
    clinicalImpact:
      'Correlates neighbouring descriptor dimensions, with a witness that proves the correlation is genuinely quantum-generated.',
  },
  readout: {
    name: '4. Sixteen Local Z Observables → Classical Head',
    type: 'Telemetry, Never the Verdict',
    math: '⟨Zⱼ⟩ (8) ⊕ ⟨ZⱼZⱼ₊₁⟩ (8) → logistic head',
    description:
      'Eight single-qubit and eight nearest-neighbour Z expectations are read from the exact Aer statevector (agreement to 8.9e-16) and fed to a small classical logistic head. The headline risk band comes from the classical baseline; this readout is reported beside it, labelled, and beside its matched classical control.',
    clinicalImpact:
      'Keeps the clinical result independent of the experimental path — if the quantum stage fails at runtime, the verdict still stands.',
  },
};

export const CircuitDiagram: React.FC = () => {
  const [selectedGate, setSelectedGate] = useState<string>('rot');

  const activeGate = GATE_DETAILS[selectedGate];
  const qubits = [0, 1, 2, 3, 4, 5, 6, 7];

  return (
    <div className="w-full space-y-4">
      {/* Interactive Quantum Circuit SVG Card */}
      <Card variant="white" padding="md" className="overflow-x-auto border-slate-200 dark:border-slate-800 shadow-xs">
        <div className="min-w-[720px] py-1">
          {/* Circuit Header / Legend */}
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-100 dark:border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              <span className="font-bold text-slate-800 dark:text-slate-100">
                8-Qubit Quantum Feature Map (Exact Qiskit Aer Statevector)
              </span>
            </div>
            <span className="text-slate-500 dark:text-slate-400 font-medium">Click any circuit block to inspect the exact mathematical operation</span>
          </div>

          <svg viewBox="0 0 780 340" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto">
            {/* Background wire lanes */}
            {qubits.map((q) => {
              const y = 35 + q * 38;
              return (
                <g key={q}>
                  {/* Qubit Label */}
                  <text
                    x="15"
                    y={y + 4}
                    fill="#64748B"
                    className="dark:fill-slate-300"
                    fontSize="12"
                    fontFamily="monospace"
                    fontWeight="700"
                  >
                    |q_{q}⟩
                  </text>

                  {/* Wire line */}
                  <line x1="55" y1={y} x2="720" y2={y} stroke="#CBD5E1" strokeWidth="1.5" className="dark:stroke-slate-700" />
                </g>
              );
            })}

            {/* BLOCK 1: PCA-8 → Encoding Angles Column */}
            <g
              className="cursor-pointer group"
              onClick={() => setSelectedGate('prep')}
            >
              <rect
                x="65"
                y="15"
                width="95"
                height="305"
                rx="12"
                fill={selectedGate === 'prep' ? '#EEF2FF' : '#F8FAFC'}
                stroke={selectedGate === 'prep' ? '#4F46E5' : '#CBD5E1'}
                strokeWidth={selectedGate === 'prep' ? '2.5' : '1.5'}
                className="transition-all duration-200 dark:fill-slate-800/80 dark:stroke-slate-700"
              />
              <text
                x="112"
                y="160"
                fill="#312E81"
                className="dark:fill-indigo-300"
                fontSize="12"
                fontWeight="700"
                textAnchor="middle"
                transform="rotate(-90 112 160)"
              >
                1. PCA-8 → θⱼ ∈ [0, π/2]
              </text>
            </g>

            {/* BLOCK 2: Layer 1 Parameterized Rotations */}
            <g
              className="cursor-pointer group"
              onClick={() => setSelectedGate('rot')}
            >
              {qubits.map((q) => {
                const y = 35 + q * 38;
                return (
                  <g key={q}>
                    <rect
                      x="180"
                      y={y - 14}
                      width="80"
                      height="28"
                      rx="6"
                      fill={selectedGate === 'rot' ? '#EDE9FE' : '#FAF5FF'}
                      stroke={selectedGate === 'rot' ? '#7C3AED' : '#DDD6FE'}
                      strokeWidth={selectedGate === 'rot' ? '2' : '1'}
                      className="transition-all duration-200 dark:fill-purple-950/60 dark:stroke-purple-800"
                    />
                    <text
                      x="220"
                      y={y + 4}
                      fill="#5B21B6"
                      className="dark:fill-purple-300"
                      fontSize="10"
                      fontWeight="700"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      Ry(θⱼ)  ·  block 1
                    </text>
                  </g>
                );
              })}
            </g>

            {/* BLOCK 3: Nearest-Neighbour CZ Ring (symmetric — both ends are control dots) */}
            <g
              className="cursor-pointer group"
              onClick={() => setSelectedGate('cnot')}
            >
              <rect
                x="280"
                y="15"
                width="120"
                height="305"
                rx="12"
                fill={selectedGate === 'cnot' ? '#F0FDFA' : '#F8FAFC'}
                stroke={selectedGate === 'cnot' ? '#0D9488' : '#CBD5E1'}
                strokeWidth={selectedGate === 'cnot' ? '2.5' : '1.5'}
                className="transition-all duration-200 dark:fill-teal-950/40 dark:stroke-teal-800"
              />
              {qubits.slice(0, 7).map((q) => {
                const y1 = 35 + q * 38;
                const y2 = 35 + (q + 1) * 38;
                const x = 305 + (q % 3) * 25;
                return (
                  <g key={q}>
                    {/* CZ is symmetric: a filled dot at each end, joined by a link */}
                    <circle cx={x} cy={y1} r="4.5" fill="#0D9488" />
                    <line x1={x} y1={y1} x2={x} y2={y2} stroke="#0D9488" strokeWidth="2" />
                    <circle cx={x} cy={y2} r="4.5" fill="#0D9488" />
                  </g>
                );
              })}
            </g>

            {/* BLOCK 4: Layer 2 Parameterized Rotations */}
            <g
              className="cursor-pointer group"
              onClick={() => setSelectedGate('rot')}
            >
              {qubits.map((q) => {
                const y = 35 + q * 38;
                return (
                  <g key={q}>
                    <rect
                      x="420"
                      y={y - 14}
                      width="80"
                      height="28"
                      rx="6"
                      fill={selectedGate === 'rot' ? '#EDE9FE' : '#FAF5FF'}
                      stroke={selectedGate === 'rot' ? '#7C3AED' : '#DDD6FE'}
                      strokeWidth={selectedGate === 'rot' ? '2' : '1'}
                      className="transition-all duration-200 dark:fill-purple-950/60 dark:stroke-purple-800"
                    />
                    <text
                      x="460"
                      y={y + 4}
                      fill="#5B21B6"
                      className="dark:fill-purple-300"
                      fontSize="10"
                      fontWeight="700"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      Ry(θⱼ)  ·  block 2
                    </text>
                  </g>
                );
              })}
            </g>

            {/* BLOCK 5: CZ Ring Closure q₇–q₀ */}
            <g
              className="cursor-pointer group"
              onClick={() => setSelectedGate('cnot')}
            >
              <rect
                x="520"
                y="15"
                width="100"
                height="305"
                rx="12"
                fill={selectedGate === 'cnot' ? '#F0FDFA' : '#F8FAFC'}
                stroke={selectedGate === 'cnot' ? '#0D9488' : '#CBD5E1'}
                strokeWidth={selectedGate === 'cnot' ? '2.5' : '1.5'}
                className="transition-all duration-200 dark:fill-teal-950/40 dark:stroke-teal-800"
              />
              <text
                x="570"
                y="160"
                fill="#0F766E"
                className="dark:fill-teal-300"
                fontSize="11"
                fontWeight="700"
                textAnchor="middle"
                transform="rotate(-90 570 160)"
              >
                CZ Ring Closure (q₇, q₀)
              </text>
            </g>

            {/* BLOCK 6: Z Measurement on Every Qubit (8 singles + 8 NN pairs = 16 observables) */}
            <g
              className="cursor-pointer group"
              onClick={() => setSelectedGate('readout')}
            >
              {qubits.map((q) => {
                const y = 35 + q * 38;
                return (
                  <g key={q}>
                    <rect
                      x="640"
                      y={y - 17}
                      width="74"
                      height="34"
                      rx="8"
                      fill={selectedGate === 'readout' ? '#FEF2F2' : '#FFFFFF'}
                      stroke={selectedGate === 'readout' ? '#DC2626' : '#EF4444'}
                      strokeWidth={selectedGate === 'readout' ? '2.5' : '1.5'}
                      className="transition-all duration-200 dark:fill-rose-950/50 dark:stroke-rose-700"
                    />
                    <path
                      d={`M 652 ${y + 5} A 10 10 0 0 1 672 ${y + 5}`}
                      stroke="#DC2626"
                      strokeWidth="1.5"
                      fill="none"
                    />
                    <line x1="662" y1={y + 5} x2="668" y2={y - 7} stroke="#DC2626" strokeWidth="1.5" />
                    <text
                      x="678"
                      y={y + 5}
                      fill="#991B1B"
                      className="dark:fill-rose-300"
                      fontSize="10"
                      fontWeight="700"
                    >
                      M(Z)
                    </text>
                  </g>
                );
              })}

              {/* Arrow out to the classical logistic head */}
              <line x1="714" y1="168" x2="736" y2="168" stroke="#DC2626" strokeWidth="2" strokeDasharray="3 3" />
              <polygon points="736,165 744,168 736,171" fill="#DC2626" />
              <text
                x="778"
                y="172"
                fill="#0F172A"
                className="dark:fill-slate-200"
                fontSize="10"
                fontWeight="700"
                fontFamily="monospace"
                textAnchor="end"
              >
                16 ⟨Z⟩
              </text>
            </g>
          </svg>
        </div>
      </Card>

      {/* Selected Gate Inspection Detail Panel */}
      <Card variant="stone" padding="md" className="border-indigo-100 dark:border-indigo-950/60 bg-gradient-to-r from-indigo-50/50 via-white to-purple-50/40 dark:from-indigo-950/30 dark:via-slate-900 dark:to-purple-950/30 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div className="space-y-2 max-w-xl">
            <div className="flex items-center gap-2">
              <Badge variant={selectedGate === 'cnot' ? 'teal' : selectedGate === 'rot' ? 'quantum' : selectedGate === 'prep' ? 'iris' : 'danger'}>
                {activeGate.type}
              </Badge>
              <h4 className="text-base font-bold text-slate-900 dark:text-white">{activeGate.name}</h4>
            </div>
            <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">{activeGate.description}</p>
            <div className="flex items-start gap-2 pt-1 text-xs text-slate-800 dark:text-slate-200 bg-white dark:bg-slate-800/80 p-2.5 rounded-xl border border-slate-200 dark:border-slate-700">
              <Info className="w-4 h-4 text-teal-700 dark:text-teal-400 shrink-0 mt-0.5" />
              <span>
                <strong className="text-slate-900 dark:text-white">Clinical Purpose: </strong>
                {activeGate.clinicalImpact}
              </span>
            </div>
          </div>

          <div className="shrink-0 bg-white dark:bg-slate-800 px-4 py-2.5 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-2xs">
            <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
              Mathematical Formulation
            </p>
            <code className="text-xs font-mono font-bold text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/60 px-2 py-1 rounded-lg block border border-transparent dark:border-indigo-800/50">
              {activeGate.math}
            </code>
          </div>
        </div>
      </Card>
    </div>
  );
};
