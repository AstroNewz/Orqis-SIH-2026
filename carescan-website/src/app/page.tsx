import React from 'react';
import { Navbar } from '@/components/layout/Navbar';
import { Footer } from '@/components/layout/Footer';
import { HeroSection } from '@/components/sections/HeroSection';
import { PurposeSection } from '@/components/sections/PurposeSection';
import { HowItWorks } from '@/components/sections/HowItWorks';
import { TechnologyStack } from '@/components/sections/TechnologyStack';
import { InteractiveDemo } from '@/components/sections/InteractiveDemo';
import { PhilosophySection } from '@/components/sections/PhilosophySection';
import { TimelineSection } from '@/components/sections/TimelineSection';
import { TeamSection } from '@/components/sections/TeamSection';

export default function Home() {
  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-900 dark:text-slate-100 selection:bg-teal-100 selection:text-teal-950 dark:selection:bg-teal-900 dark:selection:text-teal-100 transition-colors duration-300">
      {/* Navigation */}
      <Navbar />

      {/* Main Content Sections (Elevated Curtain with rounded bottom & depth shadow) */}
      <main className="relative z-10 bg-stone-50/90 dark:bg-slate-950 rounded-b-[36px] sm:rounded-b-[48px] md:rounded-b-[60px] curtain-main border-b border-slate-200/80 dark:border-slate-800/80 overflow-hidden">
        {/* 1. Hero Section */}
        <HeroSection />

        {/* 2. The Problem & Clinical Purpose */}
        <PurposeSection />

        {/* 3. How It Works (Capture -> Process -> Analyze -> Understand) */}
        <HowItWorks />

        {/* 4. Technology Stack Architecture */}
        <TechnologyStack />

        {/* 5. Interactive Conceptual Sandbox (Standalone Demo) */}
        <InteractiveDemo />

        {/* 6. Design Ethics & Human-Centered Philosophy */}
        <PhilosophySection />

        {/* 7. Project Evolution Timeline */}
        <TimelineSection />

        {/* 8. Multidisciplinary Research Team */}
        <TeamSection />
      </main>

      {/* Sticky Curtain Reveal Footer */}
      <div className="sticky bottom-0 z-0 w-full">
        <Footer />
      </div>
    </div>
  );
}
