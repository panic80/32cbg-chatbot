import { useCallback, useState } from 'react';
import { StorageKey } from '@/constants/storage';
import { getLocalStorageJSON, setLocalStorageJSON } from '@/utils/storage';

type InitialValue<T> = T | (() => T);

function resolveInitial<T>(initialValue: InitialValue<T>): T {
  return initialValue instanceof Function ? initialValue() : initialValue;
}

export function useLocalStorage<T>(
  key: StorageKey,
  initialValue: InitialValue<T>,
): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    const fallback = resolveInitial(initialValue);
    return getLocalStorageJSON<T>(key, fallback);
  });

  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      setStoredValue((previous) => {
        const valueToStore = value instanceof Function ? value(previous) : value;
        setLocalStorageJSON(key, valueToStore);
        return valueToStore;
      });
    },
    [key],
  );

  return [storedValue, setValue];
}
