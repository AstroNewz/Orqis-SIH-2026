'use client';

import React from 'react';
import { SectionHeader } from '../ui/SectionHeader';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { QUANTUM_CONCEPTS } from '@/lib/constants';
import { CircuitDiagram } from '../visual/CircuitDiagram';
import { OrganicDivider } from '../visual/OrganicDivider';
import { Atom, Zap, Shield, BookOpen, Sparkles } from 'lucide-react';

export const QuantumMLSection: React.FC = () => {
  return (
    <section id="quantum-ml" className="relative py-14 md:py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        {/* Section Header */}
        <SectionHeader
          badge="Held to a Demonstrator Bar"
          badgeVariant="quantum"
          title="What the Quantum Stage Does —"
          highlightText="And What It Has Not Earned"
          subtitle="An 8-qubit quantum feature map runs beside the classical baseline that actually sets the risk band. It is a measured demonstrator, not an advantage claim: every quantum configuration is scored against a matched classical control on identical patient rows, and the result is published exactly as it came out."
        />

        {/* Why Quantum Narrative Callout Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Card variant="stone" padding="md" organic="subtle" className="border-purple-200/80 bg-gradient-to-b from-purple-50/40 via-white to-white shadow-xs">
            <div className="w-9 h-9 rounded-2xl bg-purple-100 text-purple-700 flex items-center justify-center mb-3">
              <Atom className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1.5">
              A Fixed Map, Not a Trained Model
            </h3>
            <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
              Eight qubits carry the lesion descriptor after a TRAIN-fitted PCA-8 — one R<sub>y</sub> angle per component, bounded to [0, π/2]. There are zero trainable quantum gates; only the small classical head learns.
            </p>
          </Card>

          <Card variant="stone" padding="md" organic="subtle" className="border-teal-200/80 bg-gradient-to-b from-teal-50/40 via-white to-white shadow-xs">
            <div className="w-9 h-9 rounded-2xl bg-teal-100 text-teal-700 flex items-center justify-center mb-3">
              <Zap className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1.5">
              Entanglement You Can Audit
            </h3>
            <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
              A nearest-neighbour CZ ring correlates adjacent descriptor dimensions, and the correlation is measured rather than assumed — mean |C| = 0.184. A single-block ring witnesses exactly 0.000 and is rejected as carrying no quantum content.
            </p>
          </Card>

          <Card variant="stone" padding="md" organic="subtle" className="border-indigo-200/80 bg-gradient-to-b from-indigo-50/40 via-white to-white shadow-xs">
            <div className="w-9 h-9 rounded-2xl bg-indigo-100 text-indigo-700 flex items-center justify-center mb-3">
              <Shield className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1.5">
              Simulated Exactly, Claimed No Further
            </h3>
            <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
              Every quantum figure here comes from an exact Qiskit Aer statevector, agreeing to 8.9e-16 at 1.47 ms per image. No quantum hardware run has been performed, and the cost of one is reported separately — never as progress toward viability.
            </p>
          </Card>
        </div>

        {/* Interactive Circuit Diagram Visualizer */}
        <div className="space-y-3 pt-2">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-lg sm:text-xl font-bold text-slate-900">
                Interactive Quantum Circuit Visualizer
              </h3>
              <p className="text-xs text-slate-600">
                Click through the shipped feature map: ROI descriptor → PCA-8 → R<sub>y</sub> angles → CZ ring → 16 Z observables
              </p>
            </div>
            <Badge variant="quantum" pulse size="sm">
              <Sparkles className="w-3 h-3 mr-1" />
              Interactive Circuit
            </Badge>
          </div>

          <CircuitDiagram />
        </div>

        {/* 4 Core Concepts Grid */}
        <div className="space-y-5 pt-2">
          <div className="text-center max-w-2xl mx-auto">
            <h3 className="text-lg sm:text-xl font-bold text-slate-900">
              Quantum Principles Explained Simply
            </h3>
            <p className="text-xs sm:text-sm text-slate-600 mt-0.5">
              What each part of the circuit does, what it measured, and the one that told us no
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {QUANTUM_CONCEPTS.map((concept, i) => (
              <Card key={i} variant="white" padding="md" organic="subtle" className="border-slate-200 shadow-2xs">
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="quantum">{concept.badge}</Badge>
                  {concept.formula && (
                    <code className="text-[11px] font-mono font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded-md">
                      {concept.formula}
                    </code>
                  )}
                </div>
                <h4 className="text-sm sm:text-base font-bold text-slate-900 mb-1.5">{concept.title}</h4>
                <p className="text-xs sm:text-sm text-slate-700 leading-relaxed mb-2.5">
                  {concept.description}
                </p>
                <div className="flex items-start gap-2 text-xs text-slate-800 bg-slate-50 p-2.5 rounded-xl border border-slate-200/80">
                  <BookOpen className="w-4 h-4 text-purple-600 shrink-0 mt-0.5" />
                  <span>
                    <strong className="text-slate-900">Everyday Analogy: </strong>
                    {concept.analogy}
                  </span>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-12">
        <OrganicDivider position="bottom" fillColor="#FAFAF9" variant="curve-2" />
      </div>
    </section>
  );
};
