import { Router, Request, Response } from 'express';
import { validateRequest } from '../middleware/validate.js';
import { visitEventSchema } from './schemas/analyticsSchemas.js';
import { requireLogging } from '../middleware/requireLogging.js';

interface AnalyticsRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  chatLogger: import('../services/logger.js').Logger;
}

const createAnalyticsRoutes = ({ rateLimiter, chatLogger }: AnalyticsRoutesConfig) => {
  const router = Router();
  const validateVisit = validateRequest(visitEventSchema);

  router.post('/api/analytics/visit', rateLimiter, requireLogging, validateVisit, (req: Request, res: Response) => {

    const { path, referrer, sessionId, locale, title, viewport, metadata } = req.body;

    chatLogger.logVisit({
      path,
      referrer: referrer || null,
      sessionId: sessionId || null,
      locale: locale || null,
      title: title || null,
      viewport: viewport || null,
      metadata: metadata && typeof metadata === 'object' ? metadata : undefined,
      userAgent: req.get('user-agent') || undefined,
    });

    return res.status(202).json({ ok: true });
  });

  return router;
};

export default createAnalyticsRoutes;
