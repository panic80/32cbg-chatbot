import { Router, Request, Response, NextFunction } from 'express';
import { getLogger } from '../services/logger.js';
import { respondWithError } from '../utils/http.js';
import { RAG_SERVICE_URL } from '../config/constants.js';
import { toStringOrUndefined } from '../utils/validation.js';
import { requireLogging } from '../middleware/requireLogging.js';
import { validateRequest, validateQuery } from '../middleware/validate.js';
import { ragConfigUpdateSchema, analyticsVisitsQuerySchema } from './schemas/adminSchemas.js';

const logger = getLogger('routes:admin');

interface AdminRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  performanceHandler: import('express').RequestHandler;
  chatLogger: import('../services/logger.js').Logger;
}

const createAdminRoutes = ({ rateLimiter, performanceHandler, chatLogger }: AdminRoutesConfig) => {
  const router = Router();

  logger.info('Registering /api/admin/performance route');
  router.get('/performance', rateLimiter, (req: Request, res: Response, next: NextFunction) =>
    performanceHandler(req, res, next),
  );
  router.all('/performance', (_req: Request, res: Response) =>
    res.status(405).json({ error: 'Method Not Allowed' }),
  );

  // OpenRouter models endpoint - fetches available models from OpenRouter API
  logger.info('Registering /api/admin/openrouter/models route');
  router.get('/openrouter/models', rateLimiter, async (req: Request, res: Response) => {
    logger.debug('Handling GET /api/admin/openrouter/models');

    try {
      // Fetch models from OpenRouter (no auth required for models list)
      const response = await fetch('https://openrouter.ai/api/v1/models');

      if (!response.ok) {
        throw new Error(`OpenRouter API error: ${response.status}`);
      }

      interface OpenRouterModel {
        id: string;
        name?: string;
        description?: string;
        context_length: number;
        hugging_face_id?: string;
        pricing?: {
          prompt: string;
          completion: string;
        };
      }
      const data = (await response.json()) as { data: OpenRouterModel[] };

      // Filter and map models for our use case
      const models = data.data
        .filter((m) => m.id && !m.id.includes('/vision')) // Exclude vision-only models
        .map((m) => ({
          id: m.id,
          name: m.name || m.id,
          description: m.description || '',
          contextLength: m.context_length,
          isOpenSource: !!(m.hugging_face_id && m.hugging_face_id !== ''),
          pricing: m.pricing
            ? {
                prompt: m.pricing.prompt,
                completion: m.pricing.completion,
              }
            : null,
        }))
        .sort((a, b) => a.name.localeCompare(b.name));

      return res.json({
        models,
        total: models.length,
        openSourceCount: models.filter((m) => m.isOpenSource).length,
        isConfigured: !!process.env.OPENROUTER_API_KEY,
      });
    } catch (error) {
      logger.error('Failed to fetch OpenRouter models', error);
      return respondWithError(res, {
        status: 500,
        error: 'OpenRouterError',
        message: 'Failed to fetch OpenRouter models',
        logger,
        level: 'error',
      });
    }
  });

  // RAG Service Config endpoints - hot-toggle settings like HyDE
  const ADMIN_API_TOKEN = process.env.ADMIN_API_TOKEN;

  logger.info('Registering /api/admin/rag/config routes');

  // GET current RAG config
  router.get('/rag/config', rateLimiter, async (req: Request, res: Response) => {
    logger.debug('Handling GET /api/admin/rag/config');
    try {
      const headers: Record<string, string> = {};
      if (ADMIN_API_TOKEN) {
        headers['Authorization'] = `Bearer ${ADMIN_API_TOKEN}`;
      }

      const response = await fetch(`${RAG_SERVICE_URL}/api/v1/admin/config/status`, {
        headers,
      });

      if (!response.ok) {
        throw new Error(`RAG service error: ${response.status}`);
      }

      const data = await response.json();
      return res.json(data);
    } catch (error) {
      logger.error('Failed to get RAG config', error);
      return respondWithError(res, {
        status: 500,
        error: 'RAGConfigError',
        message: 'Failed to get RAG configuration',
        logger,
        level: 'error',
      });
    }
  });

  // POST update RAG config (hot-toggle)
  // Security: Added Zod validation for request body to prevent prototype pollution and invalid configs
  router.post('/rag/config', rateLimiter, validateRequest(ragConfigUpdateSchema), async (req: Request, res: Response) => {
    logger.debug('Handling POST /api/admin/rag/config', req.body);
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (ADMIN_API_TOKEN) {
        headers['Authorization'] = `Bearer ${ADMIN_API_TOKEN}`;
      }

      const response = await fetch(`${RAG_SERVICE_URL}/api/v1/admin/config/update`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          config_updates: req.body.config_updates || req.body,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`RAG service error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      logger.info('RAG config updated', { updates: req.body });
      return res.json(data);
    } catch (error) {
      logger.error('Failed to update RAG config', error);
      return respondWithError(res, {
        status: 500,
        error: 'RAGConfigUpdateError',
        message: 'Failed to update RAG configuration',
        logger,
        level: 'error',
      });
    }
  });

  logger.info('Registering admin analytics visits route');
  // Security: Added query validation for date formats and path length
  router.get('/analytics/visits', rateLimiter, validateQuery(analyticsVisitsQuerySchema), requireLogging, (req: Request, res: Response) => {
    logger.debug('Handling GET /api/admin/analytics/visits');

    const filters = {
      startAt: toStringOrUndefined(req.query.startAt),
      endAt: toStringOrUndefined(req.query.endAt),
      path: toStringOrUndefined(req.query.path),
    };

    const summary = chatLogger.getVisitSummary(filters);
    return res.json({
      data: summary,
      filters: {
        startAt: filters.startAt ?? null,
        endAt: filters.endAt ?? null,
        path: filters.path ?? null,
      },
    });
  });

  return router;
};

export default createAdminRoutes;
