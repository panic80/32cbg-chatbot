import { randomUUID } from 'crypto';
import type { Response } from 'express';
import type { Logger } from '../services/logger.js';

type LogLevel = 'info' | 'warn' | 'error' | 'debug';

interface ErrorResponseOptions {
  status?: number;
  error?: string;
  message?: string;
  logger?: Logger;
  cause?: unknown;
  details?: unknown;
  level?: LogLevel;
}

interface SerializedCause {
  name?: string;
  message?: string;
  stack?: string;
  [key: string]: unknown;
}

const serializeCause = (cause: unknown): SerializedCause | unknown | null => {
  if (cause instanceof Error) {
    return {
      name: cause.name,
      message: cause.message,
      stack: cause.stack,
    };
  }

  if (!cause || typeof cause !== 'object') {
    return cause ?? null;
  }

  return Object.fromEntries(
    Object.entries(cause as Record<string, unknown>).map(([key, value]) => [
      key,
      serializeCause(value),
    ]),
  );
};

export const createErrorResponse = ({
  status = 500,
  error = status >= 500 ? 'InternalServerError' : 'BadRequest',
  message = status >= 500 ? 'An unexpected error occurred.' : 'Invalid request.',
  logger,
  cause,
  details,
  level,
}: ErrorResponseOptions) => {
  const traceId = randomUUID();
  const logLevel = level ?? (status >= 500 ? 'error' : 'warn');

  if (logger && logger[logLevel]) {
    logger[logLevel](message, {
      traceId,
      status,
      errorCode: error,
      cause: serializeCause(cause),
      details: serializeCause(details),
    });
  }

  const body: { error: string; message: string; traceId: string; details?: unknown } = {
    error,
    message,
    traceId,
  };

  if (details !== undefined) {
    body.details = details;
  }

  return { status, body, traceId };
};

export const decodeUrlParams = (value: unknown): unknown => {
  if (value == null) {
    return value;
  }

  if (typeof value === 'string') {
    const normalized = value.replace(/\+/g, ' ');
    try {
      return decodeURIComponent(normalized);
    } catch {
      return normalized;
    }
  }

  if (Array.isArray(value)) {
    return value.map((item) => decodeUrlParams(item));
  }

  if (typeof value === 'object' && value !== null) {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        decodeUrlParams(item),
      ]),
    );
  }

  return value;
};

export const respondWithError = (res: Response, options: ErrorResponseOptions) => {
  const { status, body } = createErrorResponse(options);
  return res.status(status).json(body);
};
