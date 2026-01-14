import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchPerformanceMetrics } from './performance';

declare const global: typeof globalThis;

describe('fetchPerformanceMetrics', () => {
  beforeEach(() => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        latency: {
          answerTime: {
            mean: 1500,
            p50: 1200,
            p95: 2100,
            p75: 1800,
            p99: 2300,
            min: 900,
            max: 2500,
            rate_per_minute: 1.2,
            window_size: 25,
            recent: [{ value: 1200, timestamp: '2024-01-01T00:00:00Z' }],
          },
        },
        quality: {
          contextCoverage: { mean: 0.92 },
          contextSupport: { mean: 0.88 },
          answerToContext: { mean: 1.15 },
          hallucinationRate: { mean: 0.04 },
          answerTokens: { mean: 210 },
          sourceTokens: { mean: 240 },
          sourceCount: { mean: 3 },
          retrievalScores: {
            avg: { mean: 0.74 },
            max: { mean: 0.91 },
            min: { mean: 0.42 },
            std: { mean: 0.12 },
            gap: { mean: 0.18 },
          },
          errorRate: {
            total_requests: 10,
            failed_requests: 1,
            error_rate: 0.1,
            errors_by_type: { llm: 1 },
          },
        },
        throughput: { requestsPerMinute: 3.4, totalRequests: 120 },
        meta: { windowSize: 50, updatedAt: '2024-01-01T00:01:30Z' },
        gatewayMeta: { cached: false, fetchedAt: '2024-01-01T00:01:40Z' },
      }),
      text: async () => '',
      status: 200,
      statusText: 'OK',
    } as Response);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('maps api response to typed metrics', async () => {
    const metrics = await fetchPerformanceMetrics();

    expect(metrics.latency.answerTime.mean).toBe(1500);
    expect(metrics.latency.answerTime.recent).toHaveLength(1);
    expect(metrics.quality.contextCoverage.mean).toBe(0.92);
    expect(metrics.quality.contextSupport.mean).toBe(0.88);
    expect(metrics.quality.answerTokens.mean).toBe(210);
    expect(metrics.quality.retrievalScores.avg.mean).toBe(0.74);
    expect(metrics.quality.errorRate.totalRequests).toBe(10);
    expect(metrics.throughput.requestsPerMinute).toBe(3.4);
    expect(metrics.meta.windowSize).toBe(50);
    expect(metrics.gatewayMeta?.cached).toBe(false);
  });

  it('throws for non-ok responses', async () => {
    vi.restoreAllMocks();
    vi.spyOn(global, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Server Error',
      text: async () => 'boom',
    } as Response);

    await expect(fetchPerformanceMetrics()).rejects.toThrow('Failed to load performance metrics');
  });
});
