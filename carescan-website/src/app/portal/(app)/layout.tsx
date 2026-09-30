'use client';

import { PortalShell } from '@/components/portal/PortalShell';

export default function PortalAppLayout({ children }: { children: React.ReactNode }) {
  return <PortalShell>{children}</PortalShell>;
}
