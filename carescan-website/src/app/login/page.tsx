'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import {
  Lock,
  Mail,
  KeyRound,
  ShieldCheck,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  Activity,
  Server,
  Cpu,
  LogOut,
  Sparkles,
} from 'lucide-react';

interface ClinicUser {
  id: string;
  email: string;
  clinicId: string;
  fullName: string;
  role: string;
}

interface ClinicStats {
  clinicId: string;
  totalScreenings: number;
  bandCounts?: Record<string, number>;
}

interface HealthInfo {
  status: string;
  model_version: string;
  quantum_execution_mode: string;
  quantum_qubits: number;
}

export default function LoginPage() {
  const [email, setEmail] = useState('clinician@orqis.local');
  const [password, setPassword] = useState('qqONIZ2T4EYTwau4');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [currentUser, setCurrentUser] = useState<ClinicUser | null>(null);
  const [stats, setStats] = useState<ClinicStats | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

  // Check existing session on load
  useEffect(() => {
    const token = localStorage.getItem('carescan_jwt_token');
    if (token) {
      fetch(`${API_BASE}/api/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((res) => {
          if (res.ok) return res.json();
          throw new Error('Expired session');
        })
        .then((user: ClinicUser) => {
          setCurrentUser(user);
          loadClinicData(user.clinicId, token);
        })
        .catch(() => {
          localStorage.removeItem('carescan_jwt_token');
          localStorage.removeItem('carescan_user_profile');
        });
    }
  }, [API_BASE]);

  const loadClinicData = async (clinicId: string, token: string) => {
    try {
      const [statsRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/api/clinics/${clinicId}/stats`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`${API_BASE}/health`),
      ]);

      if (statsRes.ok) {
        setStats(await statsRes.json());
      }
      if (healthRes.ok) {
        setHealth(await healthRes.json());
      }
    } catch (e) {
      console.warn('Could not fetch clinic telemetry:', e);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMsg('');

    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Authentication failed: Invalid email or password');
      }

      const data = await res.json();
      localStorage.setItem('carescan_jwt_token', data.accessToken);
      localStorage.setItem('carescan_user_profile', JSON.stringify(data.user));
      setCurrentUser(data.user);
      await loadClinicData(data.user.clinicId, data.accessToken);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unable to connect to FastAPI backend (127.0.0.1:8000)';
      setErrorMsg(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickFill = () => {
    setEmail('clinician@orqis.local');
    setPassword('qqONIZ2T4EYTwau4');
    setErrorMsg('');
  };

  const handleLogout = () => {
    localStorage.removeItem('carescan_jwt_token');
    localStorage.removeItem('carescan_user_profile');
    setCurrentUser(null);
    setStats(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between selection:bg-teal-900 selection:text-teal-100">
      {/* Top Navbar Bar */}
      <header className="px-6 py-4 border-b border-slate-800/80 bg-slate-900/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center border border-slate-700 overflow-hidden">
              <Image
                src="/orqis-logo.png"
                alt="Orqis Logo"
                width={32}
                height={32}
                className="w-full h-full object-contain"
              />
            </div>
            <span className="font-brand text-2xl tracking-tight text-white group-hover:text-teal-400 transition-colors">
              Orqis
            </span>
            <span className="hidden sm:inline-block text-xs font-semibold px-2 py-0.5 rounded-full bg-teal-950/80 text-teal-400 border border-teal-800/60 uppercase tracking-wider">
              Clinician Portal
            </span>
          </Link>

          <Link
            href="/"
            className="flex items-center gap-1.5 text-xs sm:text-sm font-medium text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Research Hub</span>
          </Link>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex items-center justify-center p-4 sm:p-8">
        <div className="w-full max-w-md">
          {currentUser ? (
            /* Logged In Dashboard Card */
            <div className="bg-slate-900/90 border border-teal-500/40 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl space-y-6 animate-in fade-in zoom-in-95 duration-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="w-10 h-10 rounded-2xl bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 font-bold text-lg">
                    {currentUser.fullName.charAt(0)}
                  </div>
                  <div>
                    <h2 className="text-base font-bold text-white">{currentUser.fullName}</h2>
                    <p className="text-xs text-slate-400">{currentUser.email}</p>
                  </div>
                </div>
                <span className="text-[11px] font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded-full">
                  Authenticated
                </span>
              </div>

              {/* Clinic & Node Info */}
              <div className="bg-slate-950/70 border border-slate-800 rounded-2xl p-4 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Assigned Clinic:</span>
                  <span className="font-mono font-semibold text-teal-300">{currentUser.clinicId}</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Access Role:</span>
                  <span className="font-semibold text-slate-200 capitalize">{currentUser.role}</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Quantum QPU Telemetry:</span>
                  <span className="font-mono text-indigo-300">
                    {health ? `${health.quantum_execution_mode} (${health.quantum_qubits}Q)` : '8-Qubit Aer Simulator'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400">Model Deployment:</span>
                  <span className="font-mono text-emerald-400">
                    {health ? health.model_version : 'v1-handcrafted'}
                  </span>
                </div>
              </div>

              {/* Clinic Stats Snapshot */}
              {stats && (
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-slate-800/40 p-2.5 rounded-xl border border-slate-800">
                    <span className="text-[10px] text-slate-400 block uppercase font-bold">Total Scans</span>
                    <span className="text-lg font-bold font-mono text-white">{stats.totalScreenings}</span>
                  </div>
                  <div className="bg-rose-950/30 p-2.5 rounded-xl border border-rose-900/40">
                    <span className="text-[10px] text-rose-300 block uppercase font-bold">High Risk</span>
                    <span className="text-lg font-bold font-mono text-rose-400">
                      {stats.bandCounts?.['HIGH RISK'] || stats.bandCounts?.['HIGH'] || 0}
                    </span>
                  </div>
                  <div className="bg-amber-950/30 p-2.5 rounded-xl border border-amber-900/40">
                    <span className="text-[10px] text-amber-300 block uppercase font-bold">Moderate</span>
                    <span className="text-lg font-bold font-mono text-amber-400">
                      {stats.bandCounts?.['MODERATE RISK'] || stats.bandCounts?.['MODERATE'] || 0}
                    </span>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-2.5 pt-2">
                <a
                  href="http://127.0.0.1:8000/portal"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-400 hover:to-cyan-500 text-slate-950 font-bold text-sm shadow-lg shadow-teal-500/20 transition-all cursor-pointer"
                >
                  <Activity className="w-4 h-4" />
                  <span>Launch Diagnostic Screening Workstation</span>
                  <ArrowRight className="w-4 h-4" />
                </a>

                <Link
                  href="/#sandbox"
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs transition-colors"
                >
                  <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                  <span>Test Conceptual Triage Sandbox</span>
                </Link>

                <button
                  type="button"
                  onClick={handleLogout}
                  className="w-full flex items-center justify-center gap-1.5 py-2 px-3 text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-950/30 rounded-lg transition-colors cursor-pointer"
                >
                  <LogOut className="w-3.5 h-3.5" />
                  <span>Sign Out Session</span>
                </button>
              </div>
            </div>
          ) : (
            /* Login Form Card */
            <div className="bg-slate-900/90 border border-slate-800/90 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl space-y-6">
              <div className="text-center space-y-2">
                <div className="inline-flex p-3 rounded-2xl bg-teal-500/10 border border-teal-500/30 text-teal-400 mb-1">
                  <Lock className="w-6 h-6" />
                </div>
                <h1 className="font-heading text-2xl font-bold text-white tracking-tight">
                  Clinician Portal Sign In
                </h1>
                <p className="text-xs text-slate-400 max-w-sm mx-auto">
                  Enter your credentials seeded from your local environment (<code className="text-teal-400 font-mono">.env</code>) to access the clinical oral screening workstation.
                </p>
              </div>

              {errorMsg && (
                <div className="p-3.5 rounded-xl bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs flex items-start gap-2.5">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{errorMsg}</span>
                </div>
              )}

              <form onSubmit={handleLogin} className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                    <Mail className="w-3.5 h-3.5 text-slate-400" />
                    Clinician Email Address
                  </label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="clinician@orqis.local"
                    className="w-full px-3.5 py-2.5 text-sm bg-slate-950 border border-slate-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 text-white font-mono transition-colors"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                    <KeyRound className="w-3.5 h-3.5 text-slate-400" />
                    Password
                  </label>
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full px-3.5 py-2.5 text-sm bg-slate-950 border border-slate-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 text-white font-mono transition-colors"
                  />
                </div>

                {/* Auto-fill Helper Button */}
                <button
                  type="button"
                  onClick={handleQuickFill}
                  className="w-full py-2 px-3 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-dashed border-slate-700 text-slate-300 hover:text-teal-300 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                  <span>⚡ Quick-Fill .env Credentials (clinician@orqis.local)</span>
                </button>

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-400 hover:to-cyan-500 text-slate-950 font-bold text-sm shadow-lg shadow-teal-500/20 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60"
                >
                  {isLoading ? (
                    <span>Authenticating with FastAPI...</span>
                  ) : (
                    <>
                      <span>Sign In to Workstation</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </form>

              {/* Security Compliance Footer */}
              <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-500 text-center flex items-center justify-center gap-2">
                <ShieldCheck className="w-4 h-4 text-teal-500" />
                <span>Bcrypt Password Hashing • Stateless JWT Bearer Auth</span>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="px-6 py-4 border-t border-slate-900 text-center text-xs text-slate-600">
        CareScan / Orqis Hybrid Quantum-Classical Platform • SIH 2026 Problem Statement SIH26139
      </footer>
    </div>
  );
}
