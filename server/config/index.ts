// Re-export everything from config modules
export * from './constants.js';
export * from './environment.js';
export * from './security.js';

// Import for local use in this file
import {
  DEFAULT_CACHE_CLEANUP_INTERVAL_MS,
  DEFAULT_CACHE_TTL_MS,
  DEFAULT_HEALTH_CHECK_TIMEOUT_MS,
  DEFAULT_INGEST_TIMEOUT_MS,
  DEFAULT_MAPS_TIMEOUT_MS,
  DEFAULT_MAX_RETRIES,
  DEFAULT_PERFORMANCE_CACHE_MS,
  DEFAULT_PERFORMANCE_TIMEOUT_MS,
  DEFAULT_RATE_LIMIT_BURST,
  DEFAULT_RATE_LIMIT_MAX,
  DEFAULT_RATE_LIMIT_WINDOW_MS,
  DEFAULT_REQUEST_TIMEOUT_MS,
  DEFAULT_RETRY_DELAY_MS,
  DEFAULT_RAG_STREAM_TIMEOUT_MS,
  DEFAULT_SOURCES_STATS_TIMEOUT_MS,
  DEFAULT_SOURCES_TIMEOUT_MS,
  getEnvNumber,
} from './constants.js';

const toBoolean = (value: unknown, fallback = false): boolean => {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1') return true;
    if (normalized === 'false' || normalized === '0') return false;
  }
  return fallback;
};

interface GatewayConfig {
  maxRetries: number;
  requestTimeout: number;
  retryDelay: number;
  cacheEnabled: boolean;
  cacheTTL: number;
  cacheCleanupInterval: number;
  rateLimitEnabled: boolean;
  rateLimitMax: number;
  rateLimitWindow: number;
  rateLimitBurst: number;
  loggingEnabled: boolean;
  logLevel: string;
  logDir: string;
  ragServiceUrl: string;
  ingestTimeout: number;
  ingestMaxRetries: number;
  ingestRetryDelay: number;
  mapsTimeout: number;
  canadaCaUrl: string;
  healthCheckTimeout: number;
  performanceTimeout: number;
  performanceCacheMs: number;
  sourcesTimeout: number;
  sourcesStatsTimeout: number;
  ragStreamTimeout: number;
  admin: {
    user: string;
    password?: string;
    apiToken?: string;
  };
  [key: string]: unknown;
}

export const createGatewayConfig = (overrides: Partial<GatewayConfig> = {}): GatewayConfig => {
  const {
    ENABLE_CACHE,
    CACHE_TTL,
    CACHE_CLEANUP_INTERVAL,
    ENABLE_RATE_LIMIT,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW,
    RATE_LIMIT_BURST,
    ENABLE_LOGGING,
    LOG_LEVEL,
    LOG_DIR,
    RAG_SERVICE_URL,
    INGEST_TIMEOUT_MS,
    INGEST_MAX_RETRIES,
    INGEST_RETRY_DELAY_MS,
    MAPS_TIMEOUT,
    CANADA_CA_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    HEALTH_CHECK_TIMEOUT_MS,
    PERFORMANCE_TIMEOUT_MS,
    PERFORMANCE_CACHE_MS,
    SOURCES_TIMEOUT_MS,
    SOURCES_STATS_TIMEOUT_MS,
    RAG_STREAM_TIMEOUT_MS,
  } = process.env;

  const baseConfig: GatewayConfig = {
    maxRetries: getEnvNumber('MAX_RETRIES', DEFAULT_MAX_RETRIES),
    requestTimeout: getEnvNumber('REQUEST_TIMEOUT', DEFAULT_REQUEST_TIMEOUT_MS),
    retryDelay: getEnvNumber('RETRY_DELAY', DEFAULT_RETRY_DELAY_MS),

    cacheEnabled: toBoolean(ENABLE_CACHE, false),
    cacheTTL: getEnvNumber('CACHE_TTL', DEFAULT_CACHE_TTL_MS),
    cacheCleanupInterval: getEnvNumber('CACHE_CLEANUP_INTERVAL', DEFAULT_CACHE_CLEANUP_INTERVAL_MS),

    rateLimitEnabled: toBoolean(ENABLE_RATE_LIMIT, false),
    rateLimitMax: getEnvNumber('RATE_LIMIT_MAX', DEFAULT_RATE_LIMIT_MAX),
    rateLimitWindow: getEnvNumber('RATE_LIMIT_WINDOW', DEFAULT_RATE_LIMIT_WINDOW_MS),
    rateLimitBurst: getEnvNumber('RATE_LIMIT_BURST', DEFAULT_RATE_LIMIT_BURST),

    loggingEnabled: toBoolean(ENABLE_LOGGING, true),
    logLevel: LOG_LEVEL || 'info',
    logDir: LOG_DIR || '/var/log/cbthis',

    ragServiceUrl: RAG_SERVICE_URL || 'http://localhost:8000',
    ingestTimeout: getEnvNumber('INGEST_TIMEOUT_MS', DEFAULT_INGEST_TIMEOUT_MS),
    ingestMaxRetries: getEnvNumber(
      'INGEST_MAX_RETRIES',
      getEnvNumber('MAX_RETRIES', DEFAULT_MAX_RETRIES),
    ),
    ingestRetryDelay: getEnvNumber('INGEST_RETRY_DELAY_MS', DEFAULT_RETRY_DELAY_MS),

    mapsTimeout: getEnvNumber('MAPS_TIMEOUT', DEFAULT_MAPS_TIMEOUT_MS),
    canadaCaUrl:
      CANADA_CA_URL ||
      'https://www.canada.ca/en/department-national-defence/services/benefits-military/pay-pension-benefits/benefits/canadian-forces-temporary-duty-travel-instructions.html',

    healthCheckTimeout: getEnvNumber('HEALTH_CHECK_TIMEOUT_MS', DEFAULT_HEALTH_CHECK_TIMEOUT_MS),
    performanceTimeout: getEnvNumber('PERFORMANCE_TIMEOUT_MS', DEFAULT_PERFORMANCE_TIMEOUT_MS),
    performanceCacheMs: getEnvNumber('PERFORMANCE_CACHE_MS', DEFAULT_PERFORMANCE_CACHE_MS),
    sourcesTimeout: getEnvNumber('SOURCES_TIMEOUT_MS', DEFAULT_SOURCES_TIMEOUT_MS),
    sourcesStatsTimeout: getEnvNumber('SOURCES_STATS_TIMEOUT_MS', DEFAULT_SOURCES_STATS_TIMEOUT_MS),
    ragStreamTimeout: getEnvNumber('RAG_STREAM_TIMEOUT_MS', DEFAULT_RAG_STREAM_TIMEOUT_MS),

    admin: {
      user: process.env.CONFIG_PANEL_USER || 'admin',
      password: process.env.CONFIG_PANEL_PASSWORD,
      apiToken: process.env.ADMIN_API_TOKEN,
    },
  };

  return {
    ...baseConfig,
    ...overrides,
  };
};
