import type { Request, Response } from 'express';
import { respondWithError } from '../utils/http.js';

interface IngestionControllerConfig {
  validateIngestionUrl?: (url: string) => Promise<string>;
  buildSseCorsHeaders?: (origin?: string) => Record<string, string>;
  setSseHeaders?: (res: Response, headers?: Record<string, string | number>) => void;
  ragService: ReturnType<typeof import('../services/RagService.js').createRagService>;
  logger: import('../services/logger.js').Logger;
}

export const createIngestionController = ({
  validateIngestionUrl,
  buildSseCorsHeaders,
  setSseHeaders,
  ragService,
  logger,
}: IngestionControllerConfig) => {
  const scopedLogger = logger?.child ? logger.child({ scope: 'controller:ingestion' }) : logger;
  const emit = (level: string, message: string, meta?: unknown) => {
    const loggerFunc = (scopedLogger as unknown as Record<string, unknown>)[level];
    if (typeof loggerFunc === 'function') {
      (loggerFunc as Function)(message, meta);
    }
  };

  const sanitizeUrl = async (
    rawUrl: string | undefined,
    _contextMessage: string,
  ): Promise<string | null> => {
    if (!rawUrl) return null;
    if (!validateIngestionUrl) return rawUrl;

    try {
      const sanitized = await validateIngestionUrl(rawUrl);
      emit('debug', 'ingestion.urlValidated', { rawUrl, sanitized });
      return sanitized;
    } catch (error: unknown) {
      const err = error as Error;
      emit('warn', 'ingestion.urlRejected', { rawUrl, error: err.message });
      throw error;
    }
  };

  const handleIngest = async (req: Request, res: Response) => {
    const { url, content, type = 'web', metadata = {}, forceRefresh = false } = req.body ?? {};

    if (!url && !content) {
      return respondWithError(res, {
        status: 400,
        error: 'InvalidIngestionRequest',
        message: 'Either URL or content must be provided.',
        logger: scopedLogger,
        level: 'warn',
      });
    }

    let sanitizedUrl: string | null = null;
    if (url) {
      try {
        sanitizedUrl = await sanitizeUrl(url, 'ingest');
      } catch (validationError: unknown) {
        const err = validationError as Error & { statusCode?: number };
        return respondWithError(res, {
          status: err.statusCode || 400,
          error: 'InvalidIngestionUrl',
          message: err.message,
          logger: scopedLogger,
          level: 'warn',
        });
      }
    }

    try {
      const response = await ragService.ingest({
        url: sanitizedUrl as string,
        content,
        type,
        metadata,
        forceRefresh,
      });

      emit('info', 'ingestion.forwardSuccess', {
        hasUrl: Boolean(sanitizedUrl),
        hasContent: Boolean(content),
        type,
      });

      return res.json(response.data);
    } catch (error: unknown) {
      const err = error as Error & { response?: { status: number; data: Record<string, unknown> } };
      emit('error', 'ingestion.forwardFailed', {
        error: err.message,
        status: err.response?.status,
      });

      if (err.response) {
        return res.status(err.response.status).json(err.response.data);
      }

      return respondWithError(res, {
        status: 500,
        error: 'IngestionUpstreamFailure',
        message: 'Failed to ingest document.',
        logger: scopedLogger,
        cause: err,
      });
    }
  };

  const handleCanadaCaIngest = async (_req: Request, res: Response) => {
    try {
      const response = await ragService.ingestCanadaCa();
      emit('info', 'ingestion.canadaCaSuccess');
      return res.json(response.data);
    } catch (error: unknown) {
      const err = error as Error & { response?: { status: number; data: Record<string, unknown> } };
      emit('error', 'ingestion.canadaCaFailed', {
        error: err.message,
        status: err.response?.status,
      });

      if (err.response) {
        return res.status(err.response.status).json(err.response.data);
      }

      return respondWithError(res, {
        status: 500,
        error: 'CanadaCaIngestionFailure',
        message: 'Failed to ingest Canada.ca content.',
        logger: scopedLogger,
        cause: err,
      });
    }
  };

  const handleProgress = async (req: Request, res: Response) => {
    const { url } = req.query;
    const targetUrl = Array.isArray(url) ? (url[0] as string) : (url as string);

    if (!targetUrl) {
      return respondWithError(res, {
        status: 400,
        error: 'MissingUrlParameter',
        message: 'URL parameter required',
        logger: scopedLogger,
        level: 'warn',
      });
    }

    let sanitizedTargetUrl;
    try {
      sanitizedTargetUrl = await sanitizeUrl(targetUrl, 'progress');
    } catch (validationError: unknown) {
      const err = validationError as Error & { statusCode?: number };
      return respondWithError(res, {
        status: err.statusCode || 400,
        error: 'InvalidIngestionUrl',
        message: err.message,
        logger: scopedLogger,
        level: 'warn',
      });
    }

    try {
      // sanitizedTargetUrl is guaranteed string here due to sanitizeUrl logic on valid input
      const response = await ragService.getProgressStream(sanitizedTargetUrl as string);

      const corsHeaders = buildSseCorsHeaders?.(req.headers.origin) || {};
      setSseHeaders?.(res, {
        ...corsHeaders,
        'X-Accel-Buffering': 'no',
      });

      emit('info', 'ingestion.progressProxy', { url: sanitizedTargetUrl });
      response.data.pipe(res);

      req.on('close', () => {
        response.data.destroy();
      });
    } catch (error: unknown) {
      const err = error as Error & { response?: { status: number } };
      emit('error', 'ingestion.progressFailed', {
        error: err.message,
        status: err.response?.status,
      });
      return respondWithError(res, {
        status: 500,
        error: 'ProgressStreamError',
        message: 'Failed to connect to progress stream',
        logger: scopedLogger,
        cause: err,
      });
    }
  };

  return {
    handleIngest,
    handleCanadaCaIngest,
    handleProgress,
  };
};
