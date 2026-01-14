import request from 'supertest';
import { beforeAll, describe, expect, it, vi } from 'vitest';

vi.mock('@google/generative-ai', () => {
  return {
    GoogleGenerativeAI: class {
      getGenerativeModel() {
        return {
          generateContent: vi.fn().mockResolvedValue({
            response: { text: () => 'mock-response' },
          }),
        };
      }
    },
  };
});

vi.mock('openai', () => {
  class MockOpenAI {
    chat = {
      completions: {
        create: vi.fn().mockResolvedValue({
          choices: [{ message: { content: 'mock-openai' } }],
        }),
      },
    };
  }
  return { default: MockOpenAI };
});

vi.mock('@anthropic-ai/sdk', () => {
  class MockAnthropic {
    messages = {
      create: vi.fn().mockResolvedValue({ content: [{ text: 'mock-anthropic' }] }),
    };
  }
  return { default: MockAnthropic };
});

let app: import('express').Express;

describe('server/main routes', () => {
  beforeAll(async () => {
    process.env.NODE_ENV = 'test';
    process.env.ENABLE_CACHE = 'false';
    process.env.ENABLE_LOGGING = 'true';
    process.env.CONFIG_PANEL_USER = 'admin';
    process.env.CONFIG_PANEL_PASSWORD = 'buMeod98!!';
    process.env.ADMIN_API_TOKEN = 'test-admin-token';
    process.env.SKIP_SECURE_ENV = 'true';
    process.env.OPENAI_API_KEY = 'test-openai-key';
    process.env.GEMINI_API_KEY = 'test-gemini-key';
    process.env.ANTHROPIC_API_KEY = 'test-anthropic-key';
    process.env.GOOGLE_MAPS_API_KEY = 'test-maps-key';
    const module = await import('../main.js');
    app = module.default;
  });

  it('returns safe public configuration', async () => {
    const response = await request(app).get('/api/config');

    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      features: expect.objectContaining({ aiChat: true }),
      api: expect.objectContaining({ chat: '/api/v2/chat' }),
      caching: expect.objectContaining({ enabled: expect.any(Boolean) }),
    });
  });

  it('rejects chat requests without a message', async () => {
    const response = await request(app)
      .post('/api/v2/chat')
      .send({ model: 'gpt-4.1-mini', provider: 'openai' });

    expect(response.status).toBe(400);
    expect(response.body.message).toBe('Validation failed');
  });

  it('returns configuration errors when provider lacks credentials', async () => {
    const response = await request(app)
      .post('/api/v2/chat')
      .send({ message: 'Hello there', model: 'gpt-4.1-mini', provider: 'openai' });

    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      response: 'mock-openai',
      model: 'gpt-4.1-mini',
    });
  });

  it('requires admin credentials for logs API', async () => {
    const unauthorized = await request(app).get('/api/admin/chat-logs?limit=1&offset=0');

    expect(unauthorized.status).toBe(401);
    expect(unauthorized.headers['www-authenticate']).toMatch(/Basic/);

    const credentials = Buffer.from('admin:buMeod98!!').toString('base64');
    const authorized = await request(app)
      .get('/api/admin/chat-logs?limit=1&offset=0')
      .set('Authorization', `Basic ${credentials}`);

    expect(authorized.status).toBe(200);
  });

  it('locks the /config route behind authentication', async () => {
    const response = await request(app).get('/config');

    expect(response.status).toBe(401);
    expect(response.headers['www-authenticate']).toMatch(/Basic/);
  });

  it('locks the /chat/config route behind authentication', async () => {
    const response = await request(app).get('/chat/config');

    expect(response.status).toBe(401);
    expect(response.headers['www-authenticate']).toMatch(/Basic/);
  });

  it('returns 404 for legacy glossary endpoints', async () => {
    const listResponse = await request(app).get('/api/v2/glossary/');
    expect(listResponse.status).toBe(404);
    expect(listResponse.body).toMatchObject({
      error: expect.stringMatching(/not found/i),
    });

    const updateResponse = await request(app)
      .post('/api/v2/glossary/update')
      .send({ glossary: {} });
    expect(updateResponse.status).toBe(404);
    expect(updateResponse.body).toMatchObject({
      error: expect.stringMatching(/not found/i),
    });
  });
});
