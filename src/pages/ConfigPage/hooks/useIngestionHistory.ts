import { StorageKeys } from '@/constants/storage';
import type { IngestionHistoryEntry } from '../types';
import { useStorageList } from '@/hooks/useStorageList';

const MAX_HISTORY_ENTRIES = 10;

export const useIngestionHistory = () => {
  const { items: ingestionHistory, append, clear } = useStorageList<IngestionHistoryEntry>(
    StorageKeys.ingestionHistory,
    MAX_HISTORY_ENTRIES,
    { initialValue: [] },
  );

  return {
    ingestionHistory,
    recordHistoryEntry: append,
    clearIngestionHistory: clear,
  };
};
