import { apiClient, ApiError } from '@/api/client';
import type {
  ErrorRateSummary,
  MetricSample,
  MetricStats,
  PerformanceMetrics,
  ThroughputMetrics,
  PerformanceMeta,
  GatewayMeta,
} from '@/types/performance';

import { toNumber } from '@/utils/validation';

const mapSamples = (raw: unknown): MetricSample[] => {
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .map((sample) => {
      if (!sample || typeof sample !== 'object') {
        return null;
      }
      const value = toNumber((sample as Record<string, unknown>).value, NaN);
      const timestamp = (sample as Record<string, unknown>).timestamp;
      if (!Number.isFinite(value) || typeof timestamp !== 'string') {
        return null;
      }
      return { value, timestamp };
    })
    .filter((sample): sample is MetricSample => sample !== null);
};

const mapMetric = (raw: unknown): MetricStats => {
  const source = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};

  return {
    count: toNumber(source.count),
    mean: toNumber(source.mean),
    min: toNumber(source.min),
    max: toNumber(source.max),
    p50: toNumber(source.p50),
    p75: toNumber(source.p75),
    p95: toNumber(source.p95),
    p99: toNumber(source.p99),
    ratePerMinute: toNumber(source.rate_per_minute ?? source.ratePerMinute),
    windowSize: toNumber(source.window_size ?? source.windowSize),
    recent: mapSamples(source.recent),
  };
};

const mapErrorRate = (raw: unknown): ErrorRateSummary => {
  const source = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
  return {
    totalRequests: toNumber(source.total_requests ?? source.totalRequests),
    failedRequests: toNumber(source.failed_requests ?? source.failedRequests),
    errorRate: toNumber(source.error_rate ?? source.errorRate),
    errorsByType: (source.errors_by_type ?? source.errorsByType ?? {}) as Record<string, number>,
  };
};

const mapThroughput = (raw: unknown): ThroughputMetrics => {
  if (!raw || typeof raw !== 'object') {
    return {};
  }
  const source = raw as Record<string, unknown>;
  const resolve = (value: unknown) => {
    const numeric = toNumber(value, Number.NaN);
    return Number.isNaN(numeric) ? undefined : numeric;
  };
  return {
    requestsPerMinute: resolve(source.requestsPerMinute ?? source.requests_per_minute),
    totalRequests: resolve(source.totalRequests ?? source.total_requests),
    successfulRequests: resolve(source.successfulRequests ?? source.successful_requests),
    failedRequests: resolve(source.failedRequests ?? source.failed_requests),
  };
};

const mapMeta = (raw: unknown): PerformanceMeta => {
  if (!raw || typeof raw !== 'object') {
    return {};
  }
  const source = raw as Record<string, unknown>;
  return {
    windowSize: (() => {
      const value = toNumber(source.windowSize ?? source.window_size, Number.NaN);
      return Number.isNaN(value) ? undefined : value;
    })(),
    updatedAt: typeof source.updatedAt === 'string' ? source.updatedAt : undefined,
    uptimeSeconds: (() => {
      const value = toNumber(source.uptimeSeconds ?? source.uptime_seconds, Number.NaN);
      return Number.isNaN(value) ? undefined : value;
    })(),
  };
};

const mapGatewayMeta = (raw: unknown): GatewayMeta | undefined => {
  if (!raw || typeof raw !== 'object') {
    return undefined;
  }
  const source = raw as Record<string, unknown>;
  return {
    cached: typeof source.cached === 'boolean' ? source.cached : undefined,
    fetchedAt: typeof source.fetchedAt === 'string' ? source.fetchedAt : undefined,
    ragEndpoint: typeof source.ragEndpoint === 'string' ? source.ragEndpoint : undefined,
  };
};

export interface FetchPerformanceOptions {
  signal?: AbortSignal;
  forceRefresh?: boolean;
}

export async function fetchPerformanceMetrics(
  options: FetchPerformanceOptions = {},
): Promise<PerformanceMetrics> {
  const { signal, forceRefresh = false } = options;
  const query = forceRefresh ? '?forceRefresh=true' : '';
  let payload: Record<string, unknown>;
  try {
    payload = await apiClient.getJson<Record<string, unknown>>(`/api/admin/performance${query}`, {
      signal,
      parseErrorResponse: true,
    });
  } catch (error) {
    if (error instanceof ApiError) {
      const errorData = error.data as Record<string, unknown> | null;
      const detail = typeof errorData?.message === 'string' ? errorData.message : error.statusText;
      throw new Error(`Failed to load performance metrics (${error.status}): ${detail}`);
    }
    throw error;
  }

  // Safe access using optional chaining on the typed payload
  const latency = payload.latency as Record<string, unknown> | undefined;
  const quality = payload.quality as Record<string, unknown> | undefined;
  const retrievalScores = quality?.retrievalScores as Record<string, unknown> | undefined;

  return {
    latency: {
      answerTime: mapMetric(latency?.answerTime),
      searchTime: mapMetric(latency?.searchTime),
      retrievalTime: mapMetric(latency?.retrievalTime),
      answerGeneration: mapMetric(latency?.answerGeneration),
      firstToken: mapMetric(latency?.firstToken),
    },
    quality: {
      contextCoverage: mapMetric(quality?.contextCoverage),
      contextSupport: mapMetric(quality?.contextSupport),
      answerToContext: mapMetric(quality?.answerToContext),
      hallucinationRate: mapMetric(quality?.hallucinationRate),
      answerTokens: mapMetric(quality?.answerTokens),
      sourceTokens: mapMetric(quality?.sourceTokens),
      sourceCount: mapMetric(quality?.sourceCount),
      retrievalScores: {
        avg: mapMetric(retrievalScores?.avg),
        max: mapMetric(retrievalScores?.max),
        min: mapMetric(retrievalScores?.min),
        std: mapMetric(retrievalScores?.std),
        gap: mapMetric(retrievalScores?.gap),
      },
      errorRate: mapErrorRate(quality?.errorRate),
    },
    throughput: mapThroughput(payload.throughput),
    cache: (payload.cache as Record<string, unknown>) ?? {},
    retrievers: (payload.retrievers as Record<string, unknown>) ?? {},
    tokenUsage: (payload.tokenUsage as Record<string, unknown>) ?? {},
    meta: mapMeta(payload.meta),
    gatewayMeta: mapGatewayMeta(payload.gatewayMeta),
  };
}
