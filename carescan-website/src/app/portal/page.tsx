'use client';

/**
 * `/portal` is a router, not a screen: it sends a signed-in user to their worklist
 * and everyone else to sign-in. The decision needs `sessionStorage`, so it happens
 * on the client after hydration rather than as a server redirect.
 */

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';
import { usePortalAuth } from '@/context/PortalAuthContext';

export default function PortalIndexPage() {
  const { status } = usePortalAuth();
  const router = useRouter();

  useEffect(() => {
    if (status === 'authenticated') router.replace('/portal/worklist');
    if (status === 'anonymous') router.replace('/portal/login');
  }, [status, router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-stone-50 dark:bg-slate-950">
      <div className="flex items-center gap-2.5 text-slate-500 dark:text-slate-400">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span className="text-sm font-medium">Opening the clinic portal…</span>
      </div>
    </div>
  );
}
