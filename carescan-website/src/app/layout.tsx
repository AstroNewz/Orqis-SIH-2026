import type { Metadata } from 'next';
import { DM_Serif_Display, Plus_Jakarta_Sans, Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const dmSerifDisplay = DM_Serif_Display({
  subsets: ['latin'],
  weight: ['400'],
  style: ['normal', 'italic'],
  variable: '--font-brand',
  display: 'swap',
});

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ['latin'],
  weight: ['600', '700', '800'],
  variable: '--font-heading',
  display: 'swap',
});

const inter = Inter({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-body',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['500', '700'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Orqis — Hybrid Quantum-Classical Oral Screening Research',
  description:
    'Orqis is a privacy-conscious hybrid quantum-classical research framework combining frozen MobileNet lesion localization (DEC-020), a validated classical baseline that sets every risk band, and an 8-qubit angle-encoded quantum feature map reported alongside its matched classical control. Strict patient-level evaluation integrity on the University of Peradeniya Oral Cancer Dataset v1 (SMART-OM).',
  keywords: [
    'Orqis',
    'Oral Cancer Screening',
    'Quantum Machine Learning',
    'Quantum Feature Map',
    'Matched Classical Controls',
    'MobileNet ROI Localizer',
    'Clinical Decision Support',
    'HL7 FHIR R4',
    'SMART-OM Dataset',
    'University of Peradeniya Oral Cancer Dataset',
    'Reproducible Null Results',
    'Qiskit Aer',
  ],
  authors: [{ name: 'Orqis Research Team' }],
  icons: {
    icon: '/orqis-logo.png',
  },
};

import { AppProviders } from '@/components/providers/AppProviders';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      // Smooth scrolling is wanted for in-page anchors on the marketing site, but
      // Next has to be told about it explicitly or it also animates *route*
      // transitions — so opening a worklist row glides down the new page instead
      // of starting at the top. This attribute is how Next suppresses that.
      data-scroll-behavior="smooth"
      className={`scroll-smooth ${dmSerifDisplay.variable} ${plusJakartaSans.variable} ${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body className="antialiased text-slate-900 dark:text-slate-100 bg-stone-50 dark:bg-slate-950 selection:bg-teal-100 selection:text-teal-950 dark:selection:bg-teal-900 dark:selection:text-teal-100 transition-colors duration-300">
        <AppProviders>{children}</AppProviders>
      </body>
    </html>
  );
}
