import type { Response } from 'express';

/**
 * Error response utilities for consistent error handling across chat controllers
 */

export interface ApiError {
  error: string;
  message: string;
}

export const sendConfigurationError = (res: Response, service: string): Response => {
  return res.status(500).json({
    error: 'Configuration Error',
    message: `${service} API key is not configured.`,
  } as ApiError);
};

export const sendBadRequestError = (res: Response, message: string): Response => {
  return res.status(400).json({
    error: 'Bad Request',
    message,
  } as ApiError);
};

export const sendInternalServerError = (res: Response, message: string): Response => {
  return res.status(500).json({
    error: 'Internal Server Error',
    message,
  } as ApiError);
};

export const sendRateLimitError = (res: Response): Response => {
  return res.status(429).json({
    error: 'Rate Limit Exceeded',
    message: 'Too many requests to the AI provider. Please try again later.',
  } as ApiError);
};

export const sendUnsupportedProviderError = (res: Response, provider: string): Response => {
  return res.status(400).json({
    error: 'Bad Request',
    message: `Unsupported provider: ${provider}`,
  } as ApiError);
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
