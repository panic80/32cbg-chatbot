import { beforeEach, describe, expect, it, vi } from 'vitest';
import axios from 'axios';
import performanceService from '../services/performanceService.js';

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
  },
}));

const mockAxios = axios as unknown as { get: ReturnType<typeof vi.fn> };

const samplePayload = {
  latency: {
    answerTime: {
      mean: 1200,
      count: 3,
      p50: 1100,
      p75: 1300,
      p95: 1500,
      p99: 1700,
      min: 1000,
      max: 1800,
      rate_per_minute: 2,
      window_size: 3,
      recent: [],
    },
    searchTime: { mean: 350 },
    retrievalTime: { mean: 200 },
    answerGeneration: { mean: 900 },
    firstToken: { mean: 450 },
  },
  quality: {
    contextCoverage: { mean: 0.9 },
    hallucinationRate: { mean: 0.05 },
    errorRate: {
      total_requests: 10,
      failed_requests: 1,
      error_rate: 0.1,
      errors_by_type: { llm_errors: 1 },
    },
  },
  throughput: { requestsPerMinute: 3 },
  cache: {},
  retrievers: {},
  tokenUsage: {},
  meta: { updatedAt: '2024-01-01T00:00:00Z' },
};

describe('performanceService.fetchMetrics', () => {
  beforeEach(() => {
    performanceService.clearCache();
    vi.resetAllMocks();
    process.env.RAG_SERVICE_URL = 'http://localhost:8000';
    mockAxios.get.mockResolvedValue({ data: samplePayload });
  });

  it('fetches metrics from rag-service and normalises response', async () => {
    const result = await performanceService.fetchMetrics();

    expect(mockAxios.get).toHaveBeenCalledTimes(1);
    expect(result.latency.answerTime.mean).toBe(1200);
    expect(result.quality.contextCoverage.mean).toBe(0.9);
    expect(result.gatewayMeta.cached).toBe(false);
    expect(result.gatewayMeta.ragEndpoint).toContain('/api/v1/metrics/summary');
  });

  it('returns cached metrics when within ttl', async () => {
    await performanceService.fetchMetrics();
    const cached = await performanceService.fetchMetrics();

    expect(mockAxios.get).toHaveBeenCalledTimes(1);
    expect(cached.gatewayMeta.cached).toBe(true);
  });

  it('bypasses cache when forceRefresh is true', async () => {
    await performanceService.fetchMetrics();
    await performanceService.fetchMetrics({ forceRefresh: true });

    expect(mockAxios.get).toHaveBeenCalledTimes(2);
  });
});
