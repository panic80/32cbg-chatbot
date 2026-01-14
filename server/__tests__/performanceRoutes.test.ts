import express from 'express';
import request from 'supertest';
import { describe, expect, it, vi } from 'vitest';
import createPerformanceHandler from '../routes/performance.js';

const noopRateLimiter = (_req, _res, next) => next();
const noopAdminAuth = (_req, _res, next) => next();

describe('performance routes', () => {
  it('returns metrics payload from service', async () => {
    const mockService = {
      fetchMetrics: vi.fn().mockResolvedValue({ latency: { answerTime: { mean: 123 } } }),
    };

    const app = express();
    app.get(
      '/api/admin/performance',
      noopAdminAuth,
      noopRateLimiter,
      createPerformanceHandler({ service: mockService }),
    );

    const response = await request(app).get('/api/admin/performance');

    expect(response.status).toBe(200);
    expect(response.body.latency.answerTime.mean).toBe(123);
    expect(mockService.fetchMetrics).toHaveBeenCalledWith({ forceRefresh: false });
  });

  it('propagates errors as 502', async () => {
    const mockService = {
      fetchMetrics: vi.fn().mockRejectedValue(new Error('boom')),
    };

    const app = express();
    app.get(
      '/api/admin/performance',
      noopAdminAuth,
      noopRateLimiter,
      createPerformanceHandler({ service: mockService }),
    );

    const response = await request(app).get('/api/admin/performance');

    expect(response.status).toBe(502);
    expect(response.body.error).toBe('MetricsUnavailable');
  });

  it('supports forceRefresh query parameter', async () => {
    const mockService = {
      fetchMetrics: vi.fn().mockResolvedValue({ ok: true }),
    };

    const app = express();
    app.get(
      '/api/admin/performance',
      noopAdminAuth,
      noopRateLimiter,
      createPerformanceHandler({ service: mockService }),
    );

    await request(app).get('/api/admin/performance?forceRefresh=true');

    expect(mockService.fetchMetrics).toHaveBeenCalledWith({ forceRefresh: true });
  });
});
