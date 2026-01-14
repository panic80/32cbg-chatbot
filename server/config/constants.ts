const parseEnvInt = (value: string | undefined | null, fallback: number): number => {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }

  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const DEFAULT_MAX_RETRIES = 3;
export const DEFAULT_REQUEST_TIMEOUT_MS = 10_000;
export const DEFAULT_RETRY_DELAY_MS = 1_000;

export const DEFAULT_CACHE_TTL_MS = 3_600_000; // 1 hour
export const DEFAULT_CACHE_CLEANUP_INTERVAL_MS = 300_000; // 5 minutes

export const DEFAULT_RATE_LIMIT_MAX = 60; // requests per minute
export const DEFAULT_RATE_LIMIT_WINDOW_MS = 60_000; // 1 minute
export const DEFAULT_RATE_LIMIT_BURST = 0;

export const DEFAULT_RAG_STREAM_TIMEOUT_MS = 120_000;
export const DEFAULT_MAPS_TIMEOUT_MS = 5_000;
export const DEFAULT_SOURCES_TIMEOUT_MS = 10_000;
export const DEFAULT_SOURCES_STATS_TIMEOUT_MS = 30_000;
export const DEFAULT_INGEST_TIMEOUT_MS = 300_000;
export const DEFAULT_HEALTH_CHECK_TIMEOUT_MS = 5_000;

export const DEFAULT_PERFORMANCE_TIMEOUT_MS = 7_000;
export const DEFAULT_PERFORMANCE_CACHE_MS = 5_000;

export const SERVER_DEFAULTS = {
  MAX_RETRIES: DEFAULT_MAX_RETRIES,
  REQUEST_TIMEOUT_MS: DEFAULT_REQUEST_TIMEOUT_MS,
  RETRY_DELAY_MS: DEFAULT_RETRY_DELAY_MS,
  CACHE_TTL_MS: DEFAULT_CACHE_TTL_MS,
  CACHE_CLEANUP_INTERVAL_MS: DEFAULT_CACHE_CLEANUP_INTERVAL_MS,
  RATE_LIMIT_MAX: DEFAULT_RATE_LIMIT_MAX,
  RATE_LIMIT_WINDOW_MS: DEFAULT_RATE_LIMIT_WINDOW_MS,
  RATE_LIMIT_BURST: DEFAULT_RATE_LIMIT_BURST,
  RAG_STREAM_TIMEOUT_MS: DEFAULT_RAG_STREAM_TIMEOUT_MS,
  MAPS_TIMEOUT_MS: DEFAULT_MAPS_TIMEOUT_MS,
  SOURCES_TIMEOUT_MS: DEFAULT_SOURCES_TIMEOUT_MS,
  SOURCES_STATS_TIMEOUT_MS: DEFAULT_SOURCES_STATS_TIMEOUT_MS,
  INGEST_TIMEOUT_MS: DEFAULT_INGEST_TIMEOUT_MS,
  HEALTH_CHECK_TIMEOUT_MS: DEFAULT_HEALTH_CHECK_TIMEOUT_MS,
  PERFORMANCE_TIMEOUT_MS: DEFAULT_PERFORMANCE_TIMEOUT_MS,
  PERFORMANCE_CACHE_MS: DEFAULT_PERFORMANCE_CACHE_MS,
};

export const getEnvNumber = (envKey: string, fallback: number): number =>
  parseEnvInt(process.env[envKey], fallback);

// Shared service URLs
export const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8000';

// Trip planner configuration
export const TRIP_PLANNER_MODEL = process.env.TRIP_PLANNER_MODEL || 'gpt-4.1-mini';
export const TRIP_PLANNER_PREFIX = '📋 **Trip Plan Request**';
