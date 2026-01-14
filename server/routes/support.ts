import { Router } from 'express';
import { validateRequest } from '../middleware/validate.js';
import { followUpRequestSchema } from './schemas/supportSchemas.js';
import { createSupportController } from '../controllers/supportController.js';
import { AiClients } from '../services/aiClients.js';
import { AxiosInstance } from 'axios';

interface SupportRoutesConfig extends Partial<AiClients> {
  rateLimiter: import('express').RequestHandler;
  cache: import('../services/cache.js').CacheService | null;
  config: { 
    loggingEnabled?: boolean;
    maxRetries: number;
    canadaCaUrl: string;
    requestTimeout: number;
    retryDelay: number;
  };
  processContent: (html: string) => string;
  httpClient?: AxiosInstance;
}

const createSupportRoutes = ({
  rateLimiter,
  cache,
  config,
  processContent,
  geminiClient,
  openaiClient,
  anthropicClient,
  httpClient,
}: SupportRoutesConfig) => {
  const router = Router();
  const controller = createSupportController({
    geminiClient,
    openaiClient,
    anthropicClient,
    processContent,
    cache,
    config,
    httpClient,
  });

  router.post(
    '/api/v2/followup',
    rateLimiter,
    validateRequest(followUpRequestSchema),
    controller.handleFollowUp,
  );

  router.get('/api/travel-instructions', rateLimiter, controller.handleTravelInstructions);

  return router;
};

export default createSupportRoutes;
