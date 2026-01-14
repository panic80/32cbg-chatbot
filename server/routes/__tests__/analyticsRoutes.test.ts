import express from 'express';
import request from 'supertest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import createAnalyticsRoutes from '../analytics.js';

const buildApp = ({ loggingEnabled = true } = {}) => {
  const app = express();
  app.use(express.json());

  const chatLogger = { logVisit: vi.fn() };

  process.env.ENABLE_LOGGING = loggingEnabled ? 'true' : 'false';

  app.use(
    createAnalyticsRoutes({
      rateLimiter: (_req, _res, next) => next(),
      chatLogger,
    }),
  );

  return { app, chatLogger };
};

describe('analytics routes', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('enforces payload validation', async () => {
    const { app } = buildApp();

    const response = await request(app).post('/api/analytics/visit').send({});

    expect(response.status).toBe(400);
    expect(response.body.message).toBe('Validation failed');
    expect(response.body.details?.[0]?.message).toMatch(/Path is required/i);
  });

  it('returns 503 when logging disabled', async () => {
    const { app } = buildApp({ loggingEnabled: false });

    const response = await request(app)
      .post('/api/analytics/visit')
      .send({ path: '/', metadata: { source: 'test' } });

    expect(response.status).toBe(503);
    expect(response.body.error).toBe('LoggingDisabled');
  });

  it('records visit when logging enabled', async () => {
    const { app, chatLogger } = buildApp();

    const response = await request(app)
      .post('/api/analytics/visit')
      .send({
        path: '/test',
        referrer: ' https://example.com ',
        sessionId: ' session-1 ',
        metadata: { feature: 'test' },
      });

    expect(response.status).toBe(202);
    expect(chatLogger.logVisit).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/test',
        referrer: 'https://example.com',
        sessionId: 'session-1',
        metadata: { feature: 'test' },
      }),
    );
  });
});
