import performanceService from '../services/performanceService.js';
import { getLogger } from '../services/logger.js';
import { respondWithError } from '../utils/http.js';
import type { Request, Response, NextFunction } from 'express';

const logger = getLogger('routes:performance');

interface PerformanceHandlerOptions {
  service?: typeof performanceService;
}

export const createPerformanceHandler = ({
  service = performanceService,
}: PerformanceHandlerOptions = {}) => {
  return async (req: Request, res: Response, _next: NextFunction) => {
    logger.info('Handling /api/admin/performance request');
    try {
      const forceRefresh =
        typeof req.query.forceRefresh === 'string'
          ? req.query.forceRefresh.toLowerCase() === 'true'
          : false;

      const metrics = await service.fetchMetrics({ forceRefresh });
      res.json(metrics);
    } catch (error) {
      respondWithError(res, {
        status: 502,
        error: 'MetricsUnavailable',
        message: 'Unable to retrieve RAG performance metrics at this time.',
        logger,
        cause: error,
      });
    }
  };
};

export default createPerformanceHandler;
