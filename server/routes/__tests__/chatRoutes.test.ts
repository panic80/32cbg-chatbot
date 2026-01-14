import express from 'express';
import request from 'supertest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import createChatRoutes from '../chat.js';
import { buildOpenAIParams } from '../../services/aiClients.js';

vi.mock('../../services/aiClients.js', async () => {
  const actual = await vi.importActual('../../services/aiClients.js');
  return {
    ...actual,
    buildOpenAIParams: vi.fn((model, messages) => ({ model, messages })),
  };
});

const buildApp = ({
  geminiClient = null,
  openaiClient = null,
  anthropicClient = null,
  configOverrides = {},
} = {}) => {
  const app = express();
  app.use(express.json());

  const chatLogger = { error: vi.fn(), info: vi.fn(), logChat: vi.fn() };
  const pipeStreamingResponse = vi.fn();

  const DEFAULT_RAG_STREAM_TIMEOUT_MS = 30000;
  const getEnvNumber = () => DEFAULT_RAG_STREAM_TIMEOUT_MS;
  const buildSseCorsHeaders = () => ({});
  const TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS = 'instructions';

  app.use(
    createChatRoutes({
      rateLimiter: (_req, _res, next) => next(),
      config: { loggingEnabled: false, ...configOverrides },
      chatLogger,
      getRagAuthHeaders: () => ({}),
      decodeUrlParams: (body) => body,
      aiService: {
        geminiClient,
        openaiClient,
        anthropicClient,
        buildOpenAIParams,
      },
      buildSseCorsHeaders,
      setSseHeaders: () => {},
      pipeStreamingResponse,
      getEnvNumber,
      DEFAULT_RAG_STREAM_TIMEOUT_MS,
      TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS,
    }),
  );

  return { app, buildOpenAIParams };
};

describe('/api/gemini/generateContent', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 400 when prompt is missing', async () => {
    const { app } = buildApp();

    const response = await request(app).post('/api/gemini/generateContent').send({});

    expect(response.status).toBe(400);
    expect(response.body.message).toBe('Validation failed');
    expect(response.body.details?.[0]?.message).toMatch(/Prompt is required/i);
  });

  it('returns 500 when Gemini client is not configured', async () => {
    const { app } = buildApp({ geminiClient: null });

    const response = await request(app)
      .post('/api/gemini/generateContent')
      .send({ prompt: 'Hello Gemini' });

    expect(response.status).toBe(500);
    expect(response.body.message).toMatch(/not configured/i);
  });

  it('returns generated content when Gemini client succeeds', async () => {
    const generateContent = vi.fn().mockResolvedValue({
      response: { text: () => 'mock-gemini-response' },
    });
    const mockGeminiClient = {
      getGenerativeModel: vi.fn().mockReturnValue({ generateContent }),
    };

    const { app } = buildApp({ geminiClient: mockGeminiClient });

    const response = await request(app)
      .post('/api/gemini/generateContent')
      .send({ prompt: 'Hello Gemini' });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ response: 'mock-gemini-response' });
    expect(mockGeminiClient.getGenerativeModel).toHaveBeenCalledWith({ model: 'gemini-2.5-flash' });
    expect(generateContent).toHaveBeenCalledWith('Hello Gemini');
  });
});
describe('/api/v2/chat', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 400 when message is missing', async () => {
    const { app } = buildApp();

    const response = await request(app).post('/api/v2/chat').send({ model: 'gpt-4.1-mini' });

    expect(response.status).toBe(400);
    expect(response.body.message).toBe('Validation failed');
    expect(response.body.details?.[0]?.message).toMatch(/Message is required/i);
  });

  it('returns 500 when OpenAI client is not configured', async () => {
    const { app } = buildApp();

    const response = await request(app)
      .post('/api/v2/chat')
      .send({ message: 'Hello', model: 'gpt-4.1-mini', provider: 'openai' });

    expect(response.status).toBe(500);
    expect(response.body.message).toMatch(/not configured/i);
  });

  it('returns chat response from OpenAI provider when configured', async () => {
    const createMock = vi.fn().mockResolvedValue({
      choices: [{ message: { content: 'openai-response' } }],
    });
    const openaiClient = {
      chat: {
        completions: {
          create: createMock,
        },
      },
    };

    const { app, buildOpenAIParams } = buildApp({ openaiClient });

    const response = await request(app)
      .post('/api/v2/chat')
      .send({ message: 'Hello there ', model: 'gpt-4.1-mini', provider: 'openai' });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({
      response: 'openai-response',
      sources: [],
      conversation_id: null,
      model: 'gpt-4.1-mini',
    });
    expect(buildOpenAIParams).toHaveBeenCalledWith('gpt-4.1-mini', [
      { role: 'user', content: 'Hello there' },
    ]);
    expect(createMock).toHaveBeenCalled();
  });
});
