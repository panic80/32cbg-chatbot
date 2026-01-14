import { useState, useCallback } from 'react';
import type { VisitSummary } from '@/api/analytics';
import { fetchVisitSummary } from '@/api/analytics';

export const useVisitSummary = () => {
  const [summary, setSummary] = useState<VisitSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchVisitSummary();
      setSummary(result);
      setInitialized(true);
    } catch (err) {
      setSummary(null);
      setInitialized(false);
      setError(err instanceof Error ? err.message : 'Failed to load visit summary');
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    visitSummary: summary,
    visitSummaryError: error,
    visitSummaryLoading: loading,
    visitSummaryInitialized: initialized,
    loadVisitSummary: loadSummary,
  };
};
