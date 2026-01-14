import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchPerformanceMetrics } from '@/api/performance';
import type { PerformanceMetrics, PerformanceState } from '@/types/performance';

export interface UsePerformanceMetricsOptions {
  refreshInterval?: number;
  immediate?: boolean;
}

export interface RefreshOptions {
  force?: boolean;
  silent?: boolean;
}

const DEFAULT_STATE: PerformanceState = {
  status: 'idle',
  data: null,
};

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === 'AbortError';

export function usePerformanceMetrics(options: UsePerformanceMetricsOptions = {}) {
  const { refreshInterval = 30000, immediate = true } = options;
  const [state, setState] = useState<PerformanceState>(DEFAULT_STATE);
  const controllerRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async ({ force = false, silent = false }: RefreshOptions = {}) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    if (!silent) {
      setState((prev) => ({
        status: 'loading',
        data: prev.data,
        error: undefined,
      }));
    }

    try {
      const metrics = await fetchPerformanceMetrics({
        signal: controller.signal,
        forceRefresh: force,
      });

      setState({
        status: 'success',
        data: metrics,
        error: undefined,
      });

      return metrics;
    } catch (error) {
      if (isAbortError(error)) {
        return null;
      }

      setState((prev) => ({
        status: 'error',
        data: prev.data,
        error: error instanceof Error ? error.message : 'Failed to load performance metrics',
      }));
      return null;
    }
  }, []);

  const refresh = useCallback(
    (refreshOptions: RefreshOptions = {}) => fetchData(refreshOptions),
    [fetchData],
  );

  useEffect(() => {
    if (!immediate) {
      return () => {
        controllerRef.current?.abort();
      };
    }

    fetchData().catch(() => {
      /* handled in state */
    });

    return () => {
      controllerRef.current?.abort();
    };
  }, [fetchData, immediate]);

  useEffect(() => {
    if (!refreshInterval || refreshInterval <= 0) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      fetchData({ silent: true }).catch(() => {
        /* handled */
      });
    }, refreshInterval);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [fetchData, refreshInterval]);

  const lastUpdated = useMemo(() => {
    const data = state.data;
    if (!data) {
      return undefined;
    }
    return data.gatewayMeta?.fetchedAt ?? data.meta?.updatedAt;
  }, [state.data]);

  return {
    ...state,
    isLoading: state.status === 'loading',
    isError: state.status === 'error',
    refresh,
    lastUpdated,
  } as const;
}

export default usePerformanceMetrics;
