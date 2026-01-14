/**
 * Middleware to require logging to be enabled
 * Consolidates duplicate logging check logic across routes
 */

import type { Request, Response, NextFunction } from 'express';

/**
 * Middleware that checks if analytics logging is enabled
 * Returns 503 error if ENABLE_LOGGING is not set to 'true'
 */
export function requireLogging(req: Request, res: Response, next: NextFunction): void {
  if (process.env.ENABLE_LOGGING !== 'true') {
    res.status(503).json({
      error: 'LoggingDisabled',
      message: 'Analytics logging is disabled. Enable ENABLE_LOGGING to access this endpoint.',
    });
    return;
  }

  next();
}
