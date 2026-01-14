import { useEffect, useLayoutEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { forceScrollToTop, forceScrollToTopDeferred } from '@/utils/scroll';

export default function ScrollToTop() {
  const { pathname } = useLocation();

  // Ensure browser doesn't restore scroll automatically between navigations
  useEffect(() => {
    try {
      if ('scrollRestoration' in window.history) {
        window.history.scrollRestoration = 'manual';
      }
    } catch {
      // scrollRestoration may not be supported in all browsers - fail silently
    }
  }, []);

  // Immediate scroll reset before paint
  useLayoutEffect(() => {
    forceScrollToTop();
  }, [pathname]);

  // Additional deferred scroll attempts for content-heavy pages
  useEffect(() => {
    const cleanup = forceScrollToTopDeferred();

    // Extra scroll attempt for pages that might have delayed content loading
    const extraDelay = setTimeout(() => {
      forceScrollToTop();
    }, 200);

    return () => {
      cleanup();
      clearTimeout(extraDelay);
    };
  }, [pathname]);

  return null;
}
