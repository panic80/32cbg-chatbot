import type { LogEmitter } from '../utils/loggerHelpers.js';

export interface RetryConfig {
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Initial delay in ms before first retry (default: 1000) */
  initialDelayMs?: number;
  /** Backoff multiplier - delay is multiplied by this each attempt (default: 1) */
  backoffMultiplier?: number;
  /** Maximum delay in ms between retries (default: 30000) */
  maxDelayMs?: number;
  /** HTTP status codes that should trigger a retry (default: 5xx only) */
  retryableStatusCodes?: number[];
  /** Whether to retry on timeout errors (default: true) */
  retryOnTimeout?: boolean;
  /** Optional logger emit function for retry events */
  emit?: LogEmitter;
  /** Log scope identifier for retry events */
  logScope?: string;
}

export interface RetryableError extends Error {
  response?: { status: number };
  code?: string;
}

const DEFAULT_CONFIG: Required<Omit<RetryConfig, 'emit' | 'logScope'>> = {
  maxRetries: 3,
  initialDelayMs: 1000,
  backoffMultiplier: 1,
  maxDelayMs: 30000,
  retryableStatusCodes: [500, 502, 503, 504],
  retryOnTimeout: true,
};

const wait = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

const isRetryableError = (error: unknown, config: Required<Omit<RetryConfig, 'emit' | 'logScope'>>): boolean => {
  const err = error as RetryableError;

  // Timeout errors
  if (config.retryOnTimeout && (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT')) {
    return true;
  }

  // Server errors based on status code
  const status = err.response?.status;
  if (status && config.retryableStatusCodes.includes(status)) {
    return true;
  }

  return false;
};

/**
 * Execute an operation with configurable retry logic and exponential backoff.
 *
 * @param operation - Async function to execute (receives current attempt number)
 * @param config - Retry configuration options
 * @returns Promise resolving to the operation result
 * @throws The last error encountered after all retries exhausted
 *
 * @example
 * const result = await withRetry(
 *   async (attempt) => await fetchData(url),
 *   { maxRetries: 3, initialDelayMs: 500, backoffMultiplier: 2 }
 * );
 */
export const withRetry = async <T>(
  operation: (attempt: number) => Promise<T>,
  config: RetryConfig = {},
): Promise<T> => {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config };
  const { maxRetries, initialDelayMs, backoffMultiplier, maxDelayMs, emit, logScope } = mergedConfig;

  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= Math.max(1, maxRetries); attempt += 1) {
    try {
      return await operation(attempt);
    } catch (error: unknown) {
      const err = error as RetryableError;
      lastError = err;

      const status = err.response?.status;
      emit?.('warn', `${logScope || 'retry'}.attemptFailed`, {
        attempt,
        maxRetries,
        status,
        error: err.message,
      });

      // Don't retry client errors (4xx)
      if (status && status >= 400 && status < 500) {
        throw error;
      }

      // Check if error is retryable
      if (!isRetryableError(error, mergedConfig)) {
        throw error;
      }

      // Wait before next attempt (if not last attempt)
      if (attempt < maxRetries) {
        const delay = Math.min(
          initialDelayMs * Math.pow(backoffMultiplier, attempt - 1),
          maxDelayMs,
        );
        emit?.('debug', `${logScope || 'retry'}.waiting`, { delay, nextAttempt: attempt + 1 });
        await wait(delay);
      }
    }
  }

  throw lastError;
};

/**
 * Create a retry wrapper with pre-configured defaults.
 * Useful when you want consistent retry behavior across multiple operations.
 *
 * @param defaultConfig - Default configuration to apply to all operations
 * @returns A withRetry function with the defaults pre-applied
 *
 * @example
 * const retrier = createRetryWrapper({ maxRetries: 5, emit: logger.emit });
 * const result = await retrier(() => fetchData(url));
 */
export const createRetryWrapper = (defaultConfig: RetryConfig) => {
  return <T>(
    operation: (attempt: number) => Promise<T>,
    overrideConfig: RetryConfig = {},
  ): Promise<T> => {
    return withRetry(operation, { ...defaultConfig, ...overrideConfig });
  };
};

export default { withRetry, createRetryWrapper };
