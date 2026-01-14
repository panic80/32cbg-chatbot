import { useCallback } from 'react';
import { StorageKeys } from '@/constants/storage';
import type { ActivityLogEntry } from '../types';
import { useStorageList } from '@/hooks/useStorageList';

const MAX_ACTIVITY_ENTRIES = 20;

export const useActivityLog = () => {
  const { items: activityLog, append, clear } = useStorageList<ActivityLogEntry>(
    StorageKeys.activityLog,
    MAX_ACTIVITY_ENTRIES,
    { initialValue: [] },
  );

  const appendActivityLog = useCallback(
    (action: string, details: string) => {
      const entry: ActivityLogEntry = {
        timestamp: new Date().toISOString(),
        action,
        details,
      };
      append(entry);
    },
    [append],
  );

  return {
    activityLog,
    appendActivityLog,
    clearActivityLog: clear,
  };
};
