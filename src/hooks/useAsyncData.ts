import { useState, useCallback, useEffect, useRef } from 'react';

export type AsyncStatus = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  status: AsyncStatus;
  data: T | null;
  error: string | null;
}

export interface UseAsyncDataOptions<T> {
  /** Function that fetches the data - receives AbortSignal for cancellation */
  fetcher: (signal: AbortSignal) => Promise<T>;
  /** Whether to fetch immediately on mount (default: true) */
  immediate?: boolean;
  /** Auto-refresh interval in ms (default: 0 = disabled) */
  refreshInterval?: number;
  /** Initial data value */
  initialData?: T | null;
  /** Called when fetch succeeds */
  onSuccess?: (data: T) => void;
  /** Called when fetch fails */
  onError?: (error: Error) => void;
}

export interface UseAsyncDataReturn<T> {
  /** Current data (null if not loaded) */
  data: T | null;
  /** Current loading/error status */
  status: AsyncStatus;
  /** Error message if status is 'error' */
  error: string | null;
  /** Convenience boolean for loading state */
  isLoading: boolean;
  /** Convenience boolean for error state */
  isError: boolean;
  /** Convenience boolean for success state */
  isSuccess: boolean;
  /** Manually trigger a refetch. Pass { silent: true } to not show loading state */
  refetch: (options?: { silent?: boolean }) => Promise<T | null>;
  /** Reset state to initial values */
  reset: () => void;
}

const isAbortError = (error: unknown): error is DOMException =>
  error instanceof DOMException && error.name === 'AbortError';

/**
 * Generic hook for fetching async data with loading/error states.
 * Handles cancellation, refetching, and optional auto-refresh.
 *
 * @example
 * const { data, isLoading, error, refetch } = useAsyncData({
 *   fetcher: (signal) => fetchUserProfile(userId, { signal }),
 *   immediate: true,
 * });
 */
export function useAsyncData<T>(options: UseAsyncDataOptions<T>): UseAsyncDataReturn<T> {
  const {
    fetcher,
    immediate = true,
    refreshInterval = 0,
    initialData = null,
    onSuccess,
    onError,
  } = options;

  const [state, setState] = useState<AsyncState<T>>({
    status: 'idle',
    data: initialData,
    error: null,
  });

  const controllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);

  const fetchData = useCallback(
    async ({ silent = false }: { silent?: boolean } = {}): Promise<T | null> => {
      // Cancel any in-flight request
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;

      if (!silent) {
        setState((prev) => ({
          status: 'loading',
          data: prev.data,
          error: null,
        }));
      }

      try {
        const result = await fetcher(controller.signal);

        if (!isMountedRef.current) return null;

        setState({
          status: 'success',
          data: result,
          error: null,
        });

        onSuccess?.(result);
        return result;
      } catch (error) {
        if (isAbortError(error)) {
          return null;
        }

        if (!isMountedRef.current) return null;

        const errorMessage = error instanceof Error ? error.message : 'An error occurred';

        setState((prev) => ({
          status: 'error',
          data: prev.data,
          error: errorMessage,
        }));

        if (error instanceof Error) {
          onError?.(error);
        }

        return null;
      }
    },
    [fetcher, onSuccess, onError],
  );

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    setState({
      status: 'idle',
      data: initialData,
      error: null,
    });
  }, [initialData]);

  // Initial fetch on mount
  useEffect(() => {
    isMountedRef.current = true;

    if (immediate) {
      fetchData();
    }

    return () => {
      isMountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, [fetchData, immediate]);

  // Auto-refresh interval
  useEffect(() => {
    if (!refreshInterval || refreshInterval <= 0) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      fetchData({ silent: true });
    }, refreshInterval);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [fetchData, refreshInterval]);

  return {
    data: state.data,
    status: state.status,
    error: state.error,
    isLoading: state.status === 'loading',
    isError: state.status === 'error',
    isSuccess: state.status === 'success',
    refetch: fetchData,
    reset,
  };
}

export default useAsyncData;
