import { Router, Request, Response, NextFunction } from 'express';
import { TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS } from '../constants/travelPlannerInstructions.js';
import { DEFAULT_RAG_STREAM_TIMEOUT_MS, getEnvNumber } from '../config/constants.js';
import { pipeStreamingResponse } from '../services/streaming.js';
import { createChatController } from '../controllers/chatController.js';
import { validateRequest } from '../middleware/validate.js';
import { getLogger } from '../services/logger.js';
import {
  geminiGenerationSchema,
  standardChatSchema,
  ragChatSchema,
  streamingChatSchema,
} from './schemas/chatSchemas.js';

const logger = getLogger('routes:chat');

interface ChatRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  config: { loggingEnabled?: boolean };
  chatLogger: import('../services/logger.js').Logger;
  getRagAuthHeaders: () => Record<string, string>;
  decodeUrlParams: (value: unknown) => unknown;
  aiService: import('../services/aiClients.js').AiClients & {
    buildOpenAIParams: typeof import('../services/aiClients.js').buildOpenAIParams;
  };
  buildSseCorsHeaders: (origin?: string) => Record<string, string>;
  setSseHeaders: (res: Response, headers?: Record<string, string | number>) => void;
}

const createChatRoutes = ({
  rateLimiter,
  config,
  chatLogger,
  getRagAuthHeaders,
  decodeUrlParams,
  aiService,
  buildSseCorsHeaders,
}: ChatRoutesConfig) => {
  const router = Router();
  const controller = createChatController({
    chatLogger,
    getRagAuthHeaders,
    aiService,
    config,
    pipeStreamingResponse,
    buildSseCorsHeaders,
    getEnvNumber,
    DEFAULT_RAG_STREAM_TIMEOUT_MS,
    TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS,
  });

  const validateGemini = validateRequest(geminiGenerationSchema);

  router.post('/api/chat', rateLimiter, async (req: Request, res: Response, next: NextFunction) => {
    logger.info('Legacy /api/chat endpoint called, redirecting to /api/gemini/generateContent');
    if (req.body) {
      req.body = decodeUrlParams(req.body);
    }
    if (req.body.query && !req.body.prompt) {
      req.body.prompt = req.body.query;
    }
    // Let's use the middleware as intended by Express
    return validateGemini(req, res, () => controller.handleGeminiGenerateContent(req, res));
  });

  router.post(
    '/api/gemini/generateContent',
    rateLimiter,
    validateRequest(geminiGenerationSchema),
    controller.handleGeminiGenerateContent,
  );

  router.post(
    '/api/v2/chat/rag',
    rateLimiter,
    validateRequest(ragChatSchema),
    controller.handleRagChat,
  );

  router.post(
    '/api/v2/chat',
    rateLimiter,
    validateRequest(standardChatSchema),
    controller.handleStandardChat,
  );

  router.post(
    '/api/v2/chat/stream',
    rateLimiter,
    validateRequest(streamingChatSchema),
    controller.handleStreamingChat,
  );

  return router;
};

export default createChatRoutes;
