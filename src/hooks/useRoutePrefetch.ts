import { useEffect, useRef } from 'react';

type ImportFn = () => Promise<unknown>;

type CleanupFn = () => void;

const createPrefetch = (importFn: ImportFn): CleanupFn => {
  interface Connection {
    saveData?: boolean;
    effectiveType?: string;
  }
  interface NavigatorWithConnection extends Navigator {
    connection?: Connection;
    mozConnection?: Connection;
    webkitConnection?: Connection;
  }
  const nav = typeof navigator !== 'undefined' ? (navigator as NavigatorWithConnection) : undefined;
  const connection = nav ? nav.connection || nav.mozConnection || nav.webkitConnection : undefined;

  const saveData = connection && 'saveData' in connection ? connection.saveData : false;
  const isSlowConnection =
    connection && 'effectiveType' in connection
      ? /2g/.test(connection.effectiveType as string)
      : false;

  if (saveData || isSlowConnection) {
    return () => {};
  }

  let cancelled = false;
  const runPrefetch = () => {
    if (!cancelled) {
      importFn().catch(() => {});
    }
  };

  if (typeof window === 'undefined') {
    return () => {};
  }

  const idleCallback =
    'requestIdleCallback' in window
      ? (window.requestIdleCallback as typeof window.requestIdleCallback)
      : null;
  const idleId = idleCallback
    ? idleCallback(runPrefetch, { timeout: 1500 })
    : window.setTimeout(runPrefetch, 0);

  return () => {
    cancelled = true;
    if (idleCallback && 'cancelIdleCallback' in window) {
      (window.cancelIdleCallback as typeof window.cancelIdleCallback)(idleId as number);
    } else {
      clearTimeout(idleId as number);
    }
  };
};

export const useRoutePrefetch = (importFns: ImportFn[]) => {
  const cleanupRef = useRef<CleanupFn[] | null>(null);
  const triggeredRef = useRef(false);

  useEffect(() => {
    if (typeof document === 'undefined' || importFns.length === 0) {
      return;
    }

    const startPrefetch = () => {
      if (triggeredRef.current) return;
      triggeredRef.current = true;
      cleanupRef.current = importFns.map(createPrefetch);
    };

    const interactionEvents: Array<keyof DocumentEventMap> = ['pointerdown', 'keydown'];
    interactionEvents.forEach((event) => {
      document.addEventListener(event, startPrefetch, { once: true });
    });

    return () => {
      interactionEvents.forEach((event) => {
        document.removeEventListener(event, startPrefetch);
      });
      if (cleanupRef.current) {
        cleanupRef.current.forEach((cleanup) => cleanup());
        cleanupRef.current = null;
      }
      triggeredRef.current = false;
    };
  }, [importFns]);
};

export default useRoutePrefetch;
