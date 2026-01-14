export function forceScrollToTop(): void {
  try {
    // Multiple strategies for different scroll contexts
    const candidates: (Element | Window | Document | null | undefined)[] = [
      typeof document !== 'undefined' ? document.getElementById('app-scroll-root') : null,
      typeof document !== 'undefined' ? document.scrollingElement : null,
      typeof document !== 'undefined' ? document.documentElement : null,
      typeof document !== 'undefined' ? document.body : null,
      typeof window !== 'undefined' ? window : null,
      typeof document !== 'undefined' ? document : null,
    ];

    // Force immediate scroll for window
    if (typeof window !== 'undefined') {
      try {
        window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
        window.pageYOffset = 0;
      } catch {
        // Scroll API may fail in some environments - try next strategy
      }
    }

    // Force scroll for document elements
    if (typeof document !== 'undefined') {
      try {
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
      } catch {
        // Scroll property may be read-only in some contexts - try next strategy
      }
    }

    // Process all candidates
    for (const target of candidates) {
      if (!target) continue;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const t = target as any;
      if (typeof t.scrollTo === 'function') {
        try {
          t.scrollTo({ top: 0, left: 0, behavior: 'auto' });
        } catch {
          // scrollTo may fail - continue with fallbacks
        }
      }
      if ('scrollTop' in t) {
        try {
          t.scrollTop = 0;
        } catch {
          // scrollTop may be read-only - continue with fallbacks
        }
      }
      if ('pageYOffset' in t && typeof t.pageYOffset === 'number') {
        try {
          t.pageYOffset = 0;
        } catch {
          // pageYOffset may be read-only - continue with fallbacks
        }
      }
    }
  } catch {
    // Outer catch for any unexpected errors - scroll is best-effort
  }
}

export function forceScrollToTopDeferred(): () => void {
  let raf1 = 0;
  let raf2 = 0;
  let raf3 = 0;

  // Immediate attempt
  const t1 = setTimeout(forceScrollToTop, 0);
  raf1 = typeof window !== 'undefined' ? window.requestAnimationFrame(forceScrollToTop) : 0;

  // Second pass after layout
  const t2 = setTimeout(forceScrollToTop, 50);
  raf2 =
    typeof window !== 'undefined'
      ? window.requestAnimationFrame(() => {
          setTimeout(forceScrollToTop, 10);
        })
      : 0;

  // Third pass for content-heavy pages
  const t3 = setTimeout(forceScrollToTop, 200);
  raf3 =
    typeof window !== 'undefined'
      ? window.requestAnimationFrame(() => {
          setTimeout(forceScrollToTop, 100);
        })
      : 0;

  return () => {
    clearTimeout(t1);
    clearTimeout(t2);
    clearTimeout(t3);
    if (raf1) cancelAnimationFrame(raf1);
    if (raf2) cancelAnimationFrame(raf2);
    if (raf3) cancelAnimationFrame(raf3);
  };
}
