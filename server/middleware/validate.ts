import { ZodError, ZodSchema } from 'zod';
import type { Request, Response, NextFunction } from 'express';

/**
 * Middleware factory for validating request body against a Zod schema.
 * @param schema - Zod schema for validation
 * @returns Express middleware function
 */
export const validateRequest =
  (schema: ZodSchema) => (req: Request, res: Response, next: NextFunction) => {
    try {
      const parsed = schema.parse(req.body ?? {});
      req.body = parsed;
      return next();
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Validation failed',
          details: error.issues.map((issue) => ({
            path: issue.path.join('.'),
            message: issue.message,
          })),
        });
      }

      return res.status(400).json({
        error: 'Bad Request',
        message: 'Invalid request payload.',
      });
    }
  };

/**
 * Middleware factory for validating query parameters against a Zod schema.
 * Security: Validates and sanitizes query params to prevent injection attacks.
 * @param schema - Zod schema for validation
 * @returns Express middleware function
 */
export const validateQuery =
  (schema: ZodSchema) => (req: Request, res: Response, next: NextFunction) => {
    try {
      const parsed = schema.parse(req.query ?? {});
      req.query = parsed;
      return next();
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Query validation failed',
          details: error.issues.map((issue) => ({
            path: issue.path.join('.'),
            message: issue.message,
          })),
        });
      }

      return res.status(400).json({
        error: 'Bad Request',
        message: 'Invalid query parameters.',
      });
    }
  };
