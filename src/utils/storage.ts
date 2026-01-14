import { StorageKey } from '@/constants/storage';

const isBrowser = typeof window !== 'undefined';

export const getLocalStorageItem = (key: StorageKey): string | null => {
  if (!isBrowser) {
    return null;
  }

  try {
    return window.localStorage.getItem(key);
  } catch (error) {
    console.warn(`Failed to read localStorage key "${key}"`, error);
    return null;
  }
};

export const setLocalStorageItem = (key: StorageKey, value: string): void => {
  if (!isBrowser) {
    return;
  }

  try {
    window.localStorage.setItem(key, value);
  } catch (error) {
    console.warn(`Failed to write localStorage key "${key}"`, error);
  }
};

export const removeLocalStorageItem = (key: StorageKey): void => {
  if (!isBrowser) {
    return;
  }

  try {
    window.localStorage.removeItem(key);
  } catch (error) {
    console.warn(`Failed to remove localStorage key "${key}"`, error);
  }
};

export const getLocalStorageJSON = <T>(key: StorageKey, fallback: T): T => {
  const rawValue = getLocalStorageItem(key);
  if (!rawValue) {
    return fallback;
  }

  try {
    return JSON.parse(rawValue) as T;
  } catch (error) {
    console.warn(`Failed to parse localStorage JSON for key "${key}"`, error);
    // Attempt to coerce primitive values saved without JSON.stringify
    if (typeof fallback === 'string') {
      return rawValue as unknown as T;
    }
    if (typeof fallback === 'number') {
      const coerced = Number(rawValue);
      return Number.isNaN(coerced) ? fallback : (coerced as unknown as T);
    }
    if (typeof fallback === 'boolean') {
      if (rawValue === 'true') {
        return true as unknown as T;
      }
      if (rawValue === 'false') {
        return false as unknown as T;
      }
    }
    return fallback;
  }
};

export const setLocalStorageJSON = <T>(key: StorageKey, value: T): void => {
  try {
    const serialized = JSON.stringify(value);
    setLocalStorageItem(key, serialized);
  } catch (error) {
    console.warn(`Failed to stringify JSON for key "${key}"`, error);
  }
};
