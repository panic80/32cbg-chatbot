import express, { Request, Response, NextFunction } from 'express';
import axios from 'axios';
import { getLogger } from '../services/logger.js';
import { RAG_SERVICE_URL } from '../config/constants.js';
import { getRagService, postRagService } from '../utils/ragServiceHelpers.js';

interface SourcesRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  requireAdminAuth: import('express').RequestHandler;
  getRagAuthHeaders: () => Record<string, string>;
}

export function createSourcesRoutes({
  rateLimiter,
  requireAdminAuth,
  getRagAuthHeaders,
}: SourcesRoutesConfig) {
  const router = express.Router();
  const adminMiddleware =
    typeof requireAdminAuth === 'function'
      ? requireAdminAuth
      : (req: Request, res: Response, next: NextFunction) => next();
  const buildRagAuthHeaders =
    typeof getRagAuthHeaders === 'function' ? getRagAuthHeaders : () => ({});
  const logger = getLogger('routes:sources');

  // List indexed sources
  router.get(
    '/api/v2/sources',
    adminMiddleware,
    rateLimiter,
    async (req: Request, res: Response) => {
      await getRagService(
        `${RAG_SERVICE_URL}/api/v1/sources`,
        {
          params: req.query,
          timeout: 10000,
          headers: { ...buildRagAuthHeaders() },
        },
        res,
        logger,
        'SourcesList',
        'Failed to list sources.',
      );
    },
  );

  // Get source statistics
  router.get(
    '/api/v2/sources/stats',
    adminMiddleware,
    rateLimiter,
    async (req: Request, res: Response) => {
      await getRagService(
        `${RAG_SERVICE_URL}/api/v1/sources/stats`,
        {
          timeout: 10000,
          headers: { ...buildRagAuthHeaders() },
        },
        res,
        logger,
        'SourcesStats',
        'Failed to get source statistics.',
      );
    },
  );

  // Get source count
  router.get(
    '/api/v2/sources/count',
    adminMiddleware,
    rateLimiter,
    async (req: Request, res: Response) => {
      try {
        const ragServiceUrl = RAG_SERVICE_URL;
        const ragResponse = await axios.get(`${ragServiceUrl}/api/v1/sources/count`, {
          timeout: 10000,
          headers: { ...buildRagAuthHeaders() },
        });

        res.json(ragResponse.data);
      } catch (error) {
        logger.warn('Source count error', { error });
        res.json({ count: 0, status: 'error', message: 'Unable to get count' });
      }
    },
  );

  // Purge database endpoint
  router.post(
    '/api/v2/database/purge',
    adminMiddleware,
    rateLimiter,
    async (req: Request, res: Response) => {
      logger.info('Database purge requested');
      const result = await postRagService(
        `${RAG_SERVICE_URL}/api/v1/database/purge`,
        {},
        {
          timeout: 30000,
          headers: { 'Content-Type': 'application/json', ...buildRagAuthHeaders() },
        },
        res,
        logger,
        'DatabasePurge',
        'Failed to purge database.',
      );
      if (result) {
        logger.info('Database purge completed', { result });
      }
    },
  );

  // Build BM25 index endpoint
  router.post(
    '/api/v2/database/build-bm25',
    adminMiddleware,
    rateLimiter,
    async (req: Request, res: Response) => {
      logger.info('BM25 index build requested');
      const result = await postRagService(
        `${RAG_SERVICE_URL}/api/v1/admin/bm25/rebuild`,
        {},
        {
          timeout: 30000,
          headers: { 'Content-Type': 'application/json', ...buildRagAuthHeaders() },
        },
        res,
        logger,
        'BM25Build',
        'Failed to build BM25 index.',
      );
      if (result) {
        logger.info('BM25 index build initiated', { result });
      }
    },
  );

  return router;
}

export default createSourcesRoutes;
