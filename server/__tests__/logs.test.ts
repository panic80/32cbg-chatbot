import express from 'express';
import request from 'supertest';
import createLogsRoutes from '../routes/logs.js';
import chatLogger from '../services/logger.js';

const noopRateLimiter = (_req, _res, next) => next();

describe('admin chat log routes', () => {
  const app = express();
  app.use(express.json());
  app.use(createLogsRoutes({ rateLimiter: noopRateLimiter }));

  const originalLoggingFlag = process.env.ENABLE_LOGGING;

  beforeAll(() => {
    process.env.ENABLE_LOGGING = 'true';
  });

  beforeEach(() => {
    chatLogger.clearAllLogs();
  });

  afterAll(() => {
    chatLogger.clearAllLogs();
    if (originalLoggingFlag === undefined) {
      delete process.env.ENABLE_LOGGING;
    } else {
      process.env.ENABLE_LOGGING = originalLoggingFlag;
    }
  });

  it('returns service unavailable when logging is disabled', async () => {
    process.env.ENABLE_LOGGING = 'false';

    const response = await request(app).get('/api/admin/chat-logs');

    expect(response.status).toBe(503);
    expect(response.body.error).toBe('LoggingDisabled');

    process.env.ENABLE_LOGGING = 'true';
  });

  it('returns paginated chat logs with filters', async () => {
    process.env.ENABLE_LOGGING = 'true';

    chatLogger.logChat(null, {
      timestamp: '2024-01-01T00:00:00.000Z',
      question: 'First question about travel',
      answer: 'First answer',
      model: 'gpt-4o',
      provider: 'openai',
      conversationId: 'conv-1',
      ragEnabled: true,
    });

    chatLogger.logChat(null, {
      timestamp: '2024-01-02T00:00:00.000Z',
      question: 'Second question regarding visas',
      answer: 'Second answer',
      model: 'gpt-5-mini',
      provider: 'openai',
      conversationId: 'conv-2',
      ragEnabled: false,
      shortAnswerMode: true,
    });

    const response = await request(app).get('/api/admin/chat-logs').query({
      limit: 1,
      offset: 0,
      search: 'visa',
      ragEnabled: 'false',
    });

    expect(response.status).toBe(200);
    expect(response.body.data).toHaveLength(1);
    expect(response.body.data[0].conversationId).toBe('conv-2');
    expect(response.body.pagination.hasMore).toBe(false);
    expect(response.body.filters.ragEnabled).toBe(false);
  });

  it('records visit events and surfaces summary data', async () => {
    process.env.ENABLE_LOGGING = 'true';

    const visitPayload = {
      path: '/test-page',
      sessionId: 'session-123',
      referrer: 'https://example.org',
      locale: 'en-CA',
    };

    const visitResponse = await request(app).post('/api/analytics/visit').send(visitPayload);

    expect(visitResponse.status).toBe(202);

    const summaryResponse = await request(app).get('/api/admin/analytics/visits');

    expect(summaryResponse.status).toBe(200);
    expect(summaryResponse.body.data.totalVisits).toBeGreaterThanOrEqual(1);
    expect(Array.isArray(summaryResponse.body.data.dailyCounts)).toBe(true);
    expect(summaryResponse.body.filters.path).toBeNull();
  });

  it('filters visit analytics by date window', async () => {
    process.env.ENABLE_LOGGING = 'true';

    chatLogger.logVisit({
      path: '/historic',
      metadata: { note: 'outside range' },
    });

    const summaryResponse = await request(app)
      .get('/api/admin/analytics/visits')
      .query({ startAt: '2999-01-01' });

    expect(summaryResponse.status).toBe(200);
    expect(summaryResponse.body.data.totalVisits).toBe(0);
  });

  it('returns service unavailable for visit routes when logging disabled', async () => {
    process.env.ENABLE_LOGGING = 'false';

    const writeResponse = await request(app)
      .post('/api/analytics/visit')
      .send({ path: '/disabled' });

    expect(writeResponse.status).toBe(503);

    const readResponse = await request(app).get('/api/admin/analytics/visits');

    expect(readResponse.status).toBe(503);

    process.env.ENABLE_LOGGING = 'true';
  });
});
