/**
 * AI client initialization and helpers.
 * Centralizes initialization of Gemini, OpenAI, Anthropic, and OpenRouter clients.
 */

import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { Client } from '@googlemaps/google-maps-services-js';
import { getLogger } from './logger.js';

const logger = getLogger('services:aiClients');

/**
 * Check if an API key is valid (not a placeholder).
 * @param {string} key - API key
 * @returns {boolean} True if valid
 */
export const isValidApiKey = (key: string | undefined): boolean => {
  return !!(key && !key.includes('your-') && !key.includes('-key-here') && key.length > 10);
};

/**
 * Check if a model is an O-series reasoning model.
 * @param {string} model - Model name
 * @returns {boolean} True if O-series
 */
export const isOSeriesModel = (model: string | undefined): boolean => {
  return !!(
    model &&
    (model.startsWith('o3') || model.startsWith('o4') || model === 'o1' || model === 'o1-mini')
  );
};

/**
 * Build OpenAI parameters based on model type.
 * @param {string} model - Model name
 * @param {Array} messages - Chat messages
 * @returns {Object} OpenAI API parameters
 */
export const buildOpenAIParams = (
  model: string,
  messages: import('openai/resources/chat/completions').ChatCompletionMessageParam[],
) => {
  const baseParams = {
    model: model,
    messages: messages,
  };

  const isOSeries = isOSeriesModel(model);
  logger.info(`Building OpenAI params for model: ${model}, isOSeries: ${isOSeries}`);

  if (isOSeries) {
    // O-series models only support max_completion_tokens
    return {
      ...baseParams,
      max_completion_tokens: 8192,
    };
  } else {
    // Standard models support traditional parameters
    return {
      ...baseParams,
      temperature: 0.7,
    };
  }
};

/**
 * Resolve Gemini API key from environment variables.
 * @returns {string|null} API key or null
 */
const resolveGeminiApiKey = (): string | null => {
  const primary = process.env.GEMINI_API_KEY || process.env.GOOGLE_GEMINI_API_KEY;
  if (isValidApiKey(primary)) {
    return primary as string;
  }

  if (isValidApiKey(process.env.VITE_GEMINI_API_KEY)) {
    logger.warn(
      'VITE_GEMINI_API_KEY is deprecated. Migrate to GEMINI_API_KEY to keep credentials server-side.',
    );
    return process.env.VITE_GEMINI_API_KEY as string;
  }

  return null;
};

export interface AiClients {
  geminiClient: GoogleGenerativeAI | null;
  openaiClient: OpenAI | null;
  anthropicClient: Anthropic | null;
  openrouterClient: OpenAI | null;
  googleMapsClient: Client | null;
}

/**
 * Initialize all AI clients based on environment configuration.
 * @returns {AiClients} { geminiClient, openaiClient, anthropicClient, openrouterClient, googleMapsClient }
 */
export const initializeAiClients = (): AiClients => {
  let geminiClient: GoogleGenerativeAI | null = null;
  let openaiClient: OpenAI | null = null;
  let anthropicClient: Anthropic | null = null;
  let openrouterClient: OpenAI | null = null;
  let googleMapsClient: Client | null = null;

  // Initialize Gemini
  const geminiApiKey = resolveGeminiApiKey();
  if (geminiApiKey) {
    geminiClient = new GoogleGenerativeAI(geminiApiKey);
    logger.info('Gemini API client initialized');
  } else {
    logger.info('Gemini API key not configured or invalid');
  }

  // Initialize OpenAI
  if (isValidApiKey(process.env.OPENAI_API_KEY)) {
    openaiClient = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    logger.info('OpenAI API client initialized');
  } else {
    logger.info('OpenAI API key not configured or invalid');
  }

  // Initialize Anthropic
  if (isValidApiKey(process.env.ANTHROPIC_API_KEY)) {
    anthropicClient = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    logger.info('Anthropic API client initialized');
  } else {
    logger.info('Anthropic API key not configured or invalid');
  }

  // Initialize OpenRouter (OpenAI-compatible API)
  if (isValidApiKey(process.env.OPENROUTER_API_KEY)) {
    openrouterClient = new OpenAI({
      apiKey: process.env.OPENROUTER_API_KEY,
      baseURL: 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
        'X-Title': 'CF Travel Bot',
      },
    });
    logger.info('OpenRouter API client initialized');
  } else {
    logger.info('OpenRouter API key not configured');
  }

  // Initialize Google Maps
  if (isValidApiKey(process.env.GOOGLE_MAPS_API_KEY)) {
    googleMapsClient = new Client({});
    logger.info('Google Maps API client initialized');
  } else {
    logger.info('Google Maps API key not configured or invalid');
  }

  return {
    geminiClient,
    openaiClient,
    anthropicClient,
    openrouterClient,
    googleMapsClient,
  };
};
