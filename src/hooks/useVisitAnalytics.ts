import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { sendVisitEvent } from '@/api/analytics';
import { StorageKeys } from '@/constants/storage';
import { getLocalStorageItem, setLocalStorageItem } from '@/utils/storage';

const generateSessionId = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }

    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  } catch (error) {
    console.warn('Failed to create visit session id', error);
    return null;
  }
};

const ensureSessionId = (): string | null => {
  const existing = getLocalStorageItem(StorageKeys.analyticsSessionId);
  if (existing) {
    return existing;
  }

  const created = generateSessionId();
  if (created) {
    setLocalStorageItem(StorageKeys.analyticsSessionId, created);
  }
  return created;
};

const collectViewport = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  return `${window.innerWidth}x${window.innerHeight}`;
};

const shouldTrack = (): boolean => {
  if (typeof window === 'undefined') {
    return false;
  }

  const isDev = Boolean(import.meta?.env?.DEV);
  return !isDev;
};

export default function useVisitAnalytics(): void {
  const location = useLocation();
  const lastTrackedPath = useRef<string>('');

  useEffect(() => {
    if (!shouldTrack()) {
      return;
    }

    const pathName = location?.pathname ?? '/';
    const search = location?.search ?? '';
    const combinedPath = `${pathName}${search}`;

    if (combinedPath === lastTrackedPath.current) {
      return;
    }

    lastTrackedPath.current = combinedPath;

    const sessionId = ensureSessionId();

    sendVisitEvent({
      path: combinedPath,
      referrer: document.referrer || null,
      sessionId,
      locale: typeof navigator !== 'undefined' ? navigator.language : null,
      title: typeof document !== 'undefined' ? document.title : null,
      viewport: collectViewport(),
    });
  }, [location]);
}
