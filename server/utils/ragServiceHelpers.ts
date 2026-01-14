/**
 * RAG Service helper utilities
 * Consolidates duplicate error handling and request proxy logic
 */

import type { Response } from 'express';
import type { AxiosRequestConfig, AxiosResponse } from 'axios';
import axios from 'axios';
import { respondWithError } from './http.js';
import type { Logger } from '../services/logger.js';

/**
 * Handle RAG service errors with consistent formatting
 * @param error - Error object from axios or other source
 * @param res - Express response object
 * @param logger - Logger instance
 * @param context - Context string for error identification (e.g., 'SourcesList', 'DatabasePurge')
 * @param fallbackMessage - Fallback error message if none provided
 */
export function handleRagServiceError(
  error: unknown,
  res: Response,
  logger: Logger,
  context: string,
  fallbackMessage: string,
): void {
  const err = error as Error & {
    response?: {
      status: number;
      data?: { message?: string };
    };
  };

  if (err.response) {
    // Upstream error - forward status and message
    respondWithError(res, {
      status: err.response.status,
      error: `${context}UpstreamError`,
      message: err.response.data?.message || fallbackMessage,
      logger,
      cause: err,
    });
  } else {
    // Network or other error
    respondWithError(res, {
      status: 500,
      error: `${context}Failed`,
      message: fallbackMessage,
      logger,
      cause: err,
    });
  }
}

/**
 * Proxy a request to the RAG service with error handling
 * @param endpoint - RAG service endpoint URL
 * @param options - Axios request config
 * @param res - Express response object
 * @param logger - Logger instance
 * @param context - Context string for error identification
 * @param fallbackMessage - Fallback error message
 * @returns RAG service response data or void (if error handled)
 */
export async function proxyRagServiceRequest<T = any>(
  endpoint: string,
  options: AxiosRequestConfig,
  res: Response,
  logger: Logger,
  context: string,
  fallbackMessage: string,
): Promise<T | void> {
  try {
    const ragResponse: AxiosResponse<T> = await axios.request({
      url: endpoint,
      ...options,
    });

    res.json(ragResponse.data);
    return ragResponse.data;
  } catch (error: unknown) {
    handleRagServiceError(error, res, logger, context, fallbackMessage);
  }
}

/**
 * Helper for GET requests to RAG service
 */
export async function getRagService<T = any>(
  endpoint: string,
  options: Omit<AxiosRequestConfig, 'method'>,
  res: Response,
  logger: Logger,
  context: string,
  fallbackMessage: string,
): Promise<T | void> {
  return proxyRagServiceRequest<T>(
    endpoint,
    { ...options, method: 'GET' },
    res,
    logger,
    context,
    fallbackMessage,
  );
}

/**
 * Helper for POST requests to RAG service
 */
export async function postRagService<T = any>(
  endpoint: string,
  data: any,
  options: Omit<AxiosRequestConfig, 'method' | 'data'>,
  res: Response,
  logger: Logger,
  context: string,
  fallbackMessage: string,
): Promise<T | void> {
  return proxyRagServiceRequest<T>(
    endpoint,
    { ...options, method: 'POST', data },
    res,
    logger,
    context,
    fallbackMessage,
  );
}
