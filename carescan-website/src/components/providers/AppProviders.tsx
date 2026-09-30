'use client';

import React from 'react';
import { ThemeProvider } from '@/context/ThemeContext';
import { CredentialsProvider } from '@/context/CredentialsContext';
import { BackendStatusProvider } from '@/context/BackendStatusContext';
import { CredentialsModal } from '@/components/ui/CredentialsModal';

export const AppProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ThemeProvider>
      {/* Outermost of the data providers: both the marketing sandbox and the
          clinic portal read the same live backend status from here. */}
      <BackendStatusProvider>
        <CredentialsProvider>
          {children}
          <CredentialsModal />
        </CredentialsProvider>
      </BackendStatusProvider>
    </ThemeProvider>
  );
};
