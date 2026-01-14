import express, { Request, Response, NextFunction } from 'express';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import path from 'path';
import { getLogger } from '../services/logger.js';
import { respondWithError } from '../utils/http.js';
import { validateRequest } from '../middleware/validate.js';
import { modelConfigUpdateSchema } from './schemas/modelConfigSchemas.js';

const logger = getLogger('routes:model-config');

// Allowed base directories for config file storage (security: path traversal prevention)
const ALLOWED_BASE_DIRS = [
  path.resolve(process.cwd(), 'server', 'data'),
  path.resolve(process.cwd(), 'dist-server', 'data'),
];

/**
 * Validate that a path is within allowed directories.
 * Prevents path traversal attacks via MODEL_CONFIG_PATH env var.
 * @param {string} targetPath - Path to validate
 * @returns {boolean} True if path is safe
 */
const isPathSafe = (targetPath: string): boolean => {
  const resolvedPath = path.resolve(targetPath);
  return ALLOWED_BASE_DIRS.some((baseDir) => resolvedPath.startsWith(baseDir + path.sep));
};

// Default configuration path - can be overridden via env var
const getConfigPath = (): string => {
  const customPath = process.env.MODEL_CONFIG_PATH;

  // Security: Validate custom path is within allowed directories
  if (customPath) {
    const resolvedCustomPath = path.resolve(customPath);
    if (!isPathSafe(resolvedCustomPath)) {
      logger.error(`Security: MODEL_CONFIG_PATH "${customPath}" is outside allowed directories`);
      throw new Error('MODEL_CONFIG_PATH must be within the project data directories');
    }
    return resolvedCustomPath;
  }

  // Default to data directory in project
  // In production (Docker), we use 'dist-server/data'
  // In development, we use 'server/data'
  let dataDir = path.join(process.cwd(), 'server', 'data');
  const distDataDir = path.join(process.cwd(), 'dist-server', 'data');

  if (process.env.NODE_ENV === 'production' && existsSync(path.join(process.cwd(), 'dist-server'))) {
    dataDir = distDataDir;
  }

  if (!existsSync(dataDir)) {
    try {
      mkdirSync(dataDir, { recursive: true });
    } catch (error) {
      // If we can't create it (e.g. permission error in Docker root), fall back to dist-server if it exists
      if (dataDir !== distDataDir && existsSync(distDataDir)) {
        dataDir = distDataDir;
      } else {
        throw error;
      }
    }
  }
  return path.join(dataDir, 'model-config.json');
};

interface ModelConfig {
  fastModel: { provider: string; model: string };
  smartModel: { provider: string; model: string };
  operationModels: Record<string, string>;
  updatedAt?: string;
  [key: string]: unknown;
}

// Default model configuration
const DEFAULT_CONFIG: ModelConfig = {
  fastModel: { provider: 'openai', model: 'gpt-4.1-mini' },
  smartModel: { provider: 'openai', model: 'gpt-5-mini' },
  operationModels: {
    responseGeneration: 'smart',
    hydeExpansion: 'fast',
    queryRewriting: 'fast',
    followUpGeneration: 'fast',
  },
};

/**
 * Read model config from file or return defaults
 */
const readModelConfig = (): ModelConfig => {
  const configPath = getConfigPath();
  try {
    if (existsSync(configPath)) {
      const content = readFileSync(configPath, 'utf8');
      const config = JSON.parse(content);
      // Merge with defaults to ensure all fields exist
      return {
        ...DEFAULT_CONFIG,
        ...config,
        operationModels: {
          ...DEFAULT_CONFIG.operationModels,
          ...(config.operationModels || {}),
        },
      };
    }
  } catch (error) {
    logger.warn('Failed to read model config, using defaults', error);
  }
  return DEFAULT_CONFIG;
};

/**
 * Write model config to file
 */
const writeModelConfig = (config: ModelConfig): ModelConfig => {
  const configPath = getConfigPath();
  const fullConfig = {
    ...config,
    updatedAt: new Date().toISOString(),
  };
  writeFileSync(configPath, JSON.stringify(fullConfig, null, 2));
  logger.info('Model configuration saved', { path: configPath });
  return fullConfig;
};

interface ModelConfigRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  requireAdminAuth: import('express').RequestHandler;
}

export function createModelConfigRoutes({
  rateLimiter,
  requireAdminAuth,
}: ModelConfigRoutesConfig) {
  const router = express.Router();
  const adminMiddleware =
    typeof requireAdminAuth === 'function'
      ? requireAdminAuth
      : (req: Request, res: Response, next: NextFunction) => next();

  logger.info('Registering model config routes');

  // Get current model configuration
  router.get('/api/admin/model-config', rateLimiter, (req: Request, res: Response) => {
    logger.debug('Handling GET /api/admin/model-config');
    try {
      const config = readModelConfig();
      return res.json(config);
    } catch (error) {
      return respondWithError(res, {
        status: 500,
        error: 'ModelConfigReadError',
        message: 'Failed to read model configuration',
        logger,
        cause: error,
      });
    }
  });

  // Update model configuration (requires admin auth)
  // Uses Zod schema validation for type-safe request parsing
  router.post(
    '/api/admin/model-config',
    adminMiddleware,
    rateLimiter,
    validateRequest(modelConfigUpdateSchema),
    (req: Request, res: Response) => {
      logger.debug('Handling POST /api/admin/model-config');
      try {
        // Request body is already validated by Zod schema
        const { fastModel, smartModel, operationModels } = req.body;

        const config = writeModelConfig({
          fastModel,
          smartModel,
          operationModels: operationModels || DEFAULT_CONFIG.operationModels,
        });

        logger.info('Model configuration updated', {
          fastModel: `${fastModel.provider}/${fastModel.model}`,
          smartModel: `${smartModel.provider}/${smartModel.model}`,
        });

        return res.json({ success: true, config });
      } catch (error) {
        return respondWithError(res, {
          status: 500,
          error: 'ModelConfigWriteError',
          message: 'Failed to save model configuration',
          logger,
          cause: error,
        });
      }
    },
  );

  // Reset to default configuration
  router.delete(
    '/api/admin/model-config',
    adminMiddleware,
    rateLimiter,
    (req: Request, res: Response) => {
      logger.debug('Handling DELETE /api/admin/model-config');
      try {
        const config = writeModelConfig(DEFAULT_CONFIG);
        logger.info('Model configuration reset to defaults');
        return res.json({ success: true, config });
      } catch (error) {
        return respondWithError(res, {
          status: 500,
          error: 'ModelConfigResetError',
          message: 'Failed to reset model configuration',
          logger,
          cause: error,
        });
      }
    },
  );

  return router;
}

export default createModelConfigRoutes;
