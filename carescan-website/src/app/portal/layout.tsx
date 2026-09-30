import type { Metadata } from 'next';
import { PortalAuthProvider } from '@/context/PortalAuthContext';

export const metadata: Metadata = {
  title: 'Orqis Clinic Portal',
  description:
    'Sign-in surface for clinics and hospitals running Orqis oral lesion screenings. Every risk band shown is produced by the Orqis backend.',
  // A clinical worklist has no business in a search index.
  robots: { index: false, follow: false },
};

export default function PortalLayout({ children }: { children: React.ReactNode }) {
  return <PortalAuthProvider>{children}</PortalAuthProvider>;
}
