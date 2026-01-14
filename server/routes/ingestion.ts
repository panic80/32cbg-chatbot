import { Router } from 'express';
import axios, { AxiosInstance } from 'axios';
import { validateRequest } from '../middleware/validate.js';
import { ingestionRequestSchema } from './schemas/ingestionSchemas.js';
import { getLogger } from '../services/logger.js';
import { createIngestionController } from '../controllers/ingestionController.js';
import { createRagService } from '../services/RagService.js';

interface IngestionRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  requireAdminAuth: import('express').RequestHandler;
  validateIngestionUrl: (url: string) => Promise<string>;
  getRagAuthHeaders: () => Record<string, string>;
  buildSseCorsHeaders: (origin?: string) => Record<string, string>;
  setSseHeaders: (
    res: import('express').Response,
    headers?: Record<string, string | number>,
  ) => void;
  httpClient?: AxiosInstance;
  config?: Record<string, unknown>;
}

const createIngestionRoutes = ({
  rateLimiter,
  requireAdminAuth,
  validateIngestionUrl,
  getRagAuthHeaders,
  buildSseCorsHeaders,
  setSseHeaders,
  httpClient = axios,
  config = {},
}: IngestionRoutesConfig) => {
  const router = Router();
  const logger = getLogger('routes:ingestion');

  const ragService = createRagService({
    httpClient,
    getRagAuthHeaders,
    config,
    logger,
  });

  const controller = createIngestionController({
    validateIngestionUrl,
    buildSseCorsHeaders,
    setSseHeaders,
    ragService,
    logger,
  });

  const validateIngestionPayload = validateRequest(ingestionRequestSchema);

  router.post(
    '/api/rag/ingest',
    requireAdminAuth,
    rateLimiter,
    validateIngestionPayload,
    controller.handleIngest,
  );

  router.post(
    '/api/v2/ingest',
    requireAdminAuth,
    rateLimiter,
    validateIngestionPayload,
    controller.handleIngest,
  );

  router.post(
    '/api/v2/ingest/canada-ca',
    requireAdminAuth,
    rateLimiter,
    controller.handleCanadaCaIngest,
  );

  router.get('/api/rag/ingest/progress', requireAdminAuth, controller.handleProgress);
  router.get('/api/v2/ingest/progress', requireAdminAuth, controller.handleProgress);

  return router;
};

export default createIngestionRoutes;
