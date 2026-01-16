import type { Response } from 'express';
import { respondWithError } from './http.js';
import type { Logger } from '../services/logger.js';

/**
 * Error response utilities for consistent error handling across chat controllers
 * These are convenience wrappers around the core respondWithError function.
 */

export interface ApiError {
  error: string;
  message: string;
}

export interface ErrorOptions {
  logger?: Logger;
  cause?: unknown;
}

export const sendConfigurationError = (
  res: Response,
  service: string,
  options?: ErrorOptions,
): Response => {
  return respondWithError(res, {
    status: 500,
    error: 'ConfigurationError',
    message: `${service} API key is not configured.`,
    ...options,
  });
};

export const sendBadRequestError = (
  res: Response,
  message: string,
  options?: ErrorOptions,
): Response => {
  return respondWithError(res, {
    status: 400,
    error: 'BadRequest',
    message,
    ...options,
  });
};

export const sendInternalServerError = (
  res: Response,
  message: string,
  options?: ErrorOptions,
): Response => {
  return respondWithError(res, {
    status: 500,
    error: 'InternalServerError',
    message,
    ...options,
  });
};

export const sendRateLimitError = (res: Response, options?: ErrorOptions): Response => {
  return respondWithError(res, {
    status: 429,
    error: 'RateLimitExceeded',
    message: 'Too many requests to the AI provider. Please try again later.',
    ...options,
  });
};

export const sendUnsupportedProviderError = (
  res: Response,
  provider: string,
  options?: ErrorOptions,
): Response => {
  return respondWithError(res, {
    status: 400,
    error: 'BadRequest',
    message: `Unsupported provider: ${provider}`,
    ...options,
  });
};

/**
 * Message processing utilities
 */

export interface TripPlannerConfig {
  prefix: string;
  model: string;
  provider: string;
}

export const processTripPlannerMessage = (
  message: string,
  model: string,
  provider: string,
  config: TripPlannerConfig,
): { effectiveModel: string; effectiveProvider: string; isTripPlanner: boolean } => {
  const isTripPlanner = message?.startsWith(config.prefix);
  return {
    effectiveModel: isTripPlanner ? config.model : model,
    effectiveProvider: isTripPlanner ? config.provider : provider,
    isTripPlanner,
  };
};

/**
 * Validation utilities
 */

export const validateMessage = (message: unknown): message is string => {
  return typeof message === 'string' && message.trim().length > 0;
};

export const validateModel = (model: unknown): model is string => {
  return typeof model === 'string' && model.length > 0;
};

export const validateProvider = (provider: unknown): provider is string => {
  return typeof provider === 'string' && provider.length > 0;
};
