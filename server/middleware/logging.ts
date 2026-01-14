import type { Request, Response, NextFunction } from 'express';

export const loggingMiddleware = (req: Request, res: Response, next: NextFunction) => {
  // Store the original send method
  const originalSend = res.send;

  // Override send method to capture responses
  res.send = function (this: Response, body?: unknown): Response {
    res.locals.responseData = body;
    return originalSend.call(this, body);
  };

  // Continue to next middleware
  next();
};
