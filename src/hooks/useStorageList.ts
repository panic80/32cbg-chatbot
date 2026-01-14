/**
 * Generic hook for managing a list in localStorage
 * Consolidates duplicate patterns from useActivityLog and useIngestionHistory
 */

import { useState, useEffect, useCallback } from 'react';
import {
  getLocalStorageJSON,
  removeLocalStorageItem,
  setLocalStorageJSON,
} from '@/utils/storage';

export interface UseStorageListOptions<T> {
  /**
   * Initial value if storage is empty or invalid
   */
  initialValue?: T[];
  /**
   * Custom serialization (defaults to identity function)
   */
  serialize?: (item: T) => any;
  /**
   * Custom deserialization (defaults to identity function)
   */
  deserialize?: (item: any) => T;
  /**
   * Validate that loaded data is an array of correct type
   */
  validate?: (item: unknown) => item is T;
}

export interface UseStorageListReturn<T> {
  /**
   * Current list items
   */
  items: T[];
  /**
   * Loading state (true while initial load is in progress)
   */
  loading: boolean;
  /**
   * Add an item to the beginning of the list
   */
  append: (item: T) => void;
  /**
   * Clear all items from the list
   */
  clear: () => void;
  /**
   * Set the entire list (useful for bulk operations)
   */
  setItems: (items: T[]) => void;
}

/**
 * Hook for managing a list in localStorage with max entries
 *
 * @param storageKey - localStorage key to use
 * @param maxEntries - Maximum number of entries to keep (oldest are removed)
 * @param options - Optional configuration
 * @returns Storage list state and manipulation functions
 *
 * @example
 * ```tsx
 * interface LogEntry {
 *   timestamp: string;
 *   message: string;
 * }
 *
 * const { items, append, clear } = useStorageList<LogEntry>(
 *   'my-logs',
 *   50,
 *   { initialValue: [] }
 * );
 *
 * // Add entry
 * append({ timestamp: new Date().toISOString(), message: 'Hello' });
 *
 * // Clear all
 * clear();
 * ```
 */
export function useStorageList<T>(
  storageKey: string,
  maxEntries: number,
  options: UseStorageListOptions<T> = {},
): UseStorageListReturn<T> {
  const {
    initialValue = [],
    serialize = (item) => item,
    deserialize = (item) => item as T,
    validate,
  } = options;

  const [items, setItemsState] = useState<T[]>(initialValue);
  const [loading, setLoading] = useState(true);

  // Load initial data from localStorage
  useEffect(() => {
    try {
      const saved = getLocalStorageJSON<unknown>(storageKey, initialValue);

      if (Array.isArray(saved)) {
        const deserialized = saved.map(deserialize);

        // If validation is provided, filter out invalid items
        if (validate) {
          const validated = deserialized.filter((item): item is T => validate(item));
          setItemsState(validated);
        } else {
          setItemsState(deserialized);
        }
      } else {
        setItemsState(initialValue);
      }
    } catch (error) {
      console.warn(`Failed to load ${storageKey} from localStorage:`, error);
      setItemsState(initialValue);
    } finally {
      setLoading(false);
    }
  }, [storageKey]); // Intentionally omit other deps - only reload if key changes

  // Append item to beginning of list
  const append = useCallback(
    (item: T) => {
      setItemsState((previous) => {
        const next = [item, ...previous].slice(0, maxEntries);
        const serialized = next.map(serialize);
        setLocalStorageJSON(storageKey, serialized);
        return next;
      });
    },
    [storageKey, maxEntries, serialize],
  );

  // Clear all items
  const clear = useCallback(() => {
    setItemsState(initialValue);
    removeLocalStorageItem(storageKey);
  }, [storageKey, initialValue]);

  // Set items (for bulk operations)
  const setItems = useCallback(
    (newItems: T[]) => {
      const limited = newItems.slice(0, maxEntries);
      setItemsState(limited);
      const serialized = limited.map(serialize);
      setLocalStorageJSON(storageKey, serialized);
    },
    [storageKey, maxEntries, serialize],
  );

  return {
    items,
    loading,
    append,
    clear,
    setItems,
  };
}
