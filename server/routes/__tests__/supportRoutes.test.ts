import express from 'express';
import request from 'supertest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import createSupportRoutes from '../support.js';

const buildApp = ({ geminiClient, cache, httpClient } = {}) => {
  const app = express();
  app.use(express.json());

  const processContent = vi.fn((html) => `processed:${html}`);
  const config = {
    canadaCaUrl: 'https://example.com/travel',
    maxRetries: 1,
    retryDelay: 10,
    requestTimeout: 1000,
  };

  const client = httpClient ?? { get: vi.fn() };

  app.use(
    createSupportRoutes({
      rateLimiter: (_req, _res, next) => next(),
      cache,
      config,
      processContent,
      geminiClient,
      openaiClient: null,
      anthropicClient: null,
      httpClient: client,
    }),
  );

  return { app, processContent, httpClient: client };
};

describe('support routes', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('validates follow-up payload', async () => {
    const { app } = buildApp();

    const response = await request(app).post('/api/v2/followup').send({});

    expect(response.status).toBe(400);
    expect(response.body.message).toBe('Validation failed');
    expect(response.body.details?.[0]?.message).toMatch(/User question is required/i);
  });

  it('returns fallback follow-up questions when provider unavailable', async () => {
    const { app } = buildApp();

    const response = await request(app).post('/api/v2/followup').send({
      userQuestion: 'What should I do?',
      aiResponse: 'You can explore options.',
      provider: 'openai',
    });

    expect(response.status).toBe(200);
    expect(response.body.followUpQuestions).toHaveLength(2);
  });

  it('returns cached travel instructions when available', async () => {
    const cacheStore = new Map();
    cacheStore.set('travel-instructions', {
      content: 'cached-content',
      timestamp: Date.now() - 1000,
      etag: '"etag-1"',
      lastModified: new Date().toUTCString(),
    });

    const cache = {
      get: vi.fn((key) => cacheStore.get(key)),
      set: vi.fn(),
    };

    const { app } = buildApp({ cache });

    const response = await request(app).get('/api/travel-instructions');

    expect(response.status).toBe(200);
    expect(response.body.fresh).toBe(false);
    expect(response.body.content).toBe('cached-content');
  });

  it('fetches travel instructions when cache empty', async () => {
    const cache = {
      get: vi.fn().mockResolvedValue(null),
      set: vi.fn(),
    };

    const httpClient = { get: vi.fn() };
    httpClient.get.mockResolvedValueOnce({
      status: 200,
      data: '<html>content</html>',
      headers: { 'last-modified': 'Mon, 28 Oct 2025 12:00:00 GMT' },
    });

    const { app, processContent, httpClient: client } = buildApp({ cache, httpClient });

    const response = await request(app).get('/api/travel-instructions');

    expect(response.status).toBe(200);
    expect(processContent).toHaveBeenCalled();
    expect(cache.set).toHaveBeenCalled();
    expect(response.body.fresh).toBe(true);
    expect(client.get).toHaveBeenCalledWith(
      'https://example.com/travel',
      expect.objectContaining({
        timeout: expect.any(Number),
        headers: expect.objectContaining({ Accept: expect.any(String) }),
      }),
    );
  });
});
