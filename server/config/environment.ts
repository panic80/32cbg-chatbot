import { existsSync, readFileSync } from 'fs';
import dotenv from 'dotenv';
import { getLogger } from '../services/logger.js';

const SECURE_ENV_PATH = '/etc/cbthis/env';
let hasLoaded = false;
const logger = getLogger('config:environment');

const loadSecureEnvFile = (secureEnvPath = SECURE_ENV_PATH) => {
  if (!existsSync(secureEnvPath)) {
    logger.warn('Secure environment file not found', { secureEnvPath });
    return;
  }

  try {
    const secureEnv = readFileSync(secureEnvPath, 'utf8');
    secureEnv.split('\n').forEach((line) => {
      if (line.startsWith('#') || !line.trim()) return;

      const [key, ...valueParts] = line.split('=');
      if (key && valueParts.length > 0) {
        process.env[key.trim()] = valueParts.join('=').trim();
      }
    });
    logger.info('Loaded secure environment variables', { secureEnvPath });
  } catch (error) {
    logger.error('Failed to load secure environment variables', { error });
  }
};

export const loadEnvironment = (): { nodeEnv: string } => {
  if (hasLoaded) {
    return {
      nodeEnv: process.env.NODE_ENV || 'development',
    };
  }

  hasLoaded = true;

  if (process.env.SKIP_SECURE_ENV === 'true') {
    logger.warn('Secure environment file loading skipped via SKIP_SECURE_ENV flag');
  } else {
    loadSecureEnvFile();
  }

  const nodeEnv = process.env.NODE_ENV || 'development';

  dotenv.config({ path: `.env.${nodeEnv}` });
  dotenv.config();

  return { nodeEnv };
};

export const __internal = {
  loadSecureEnvFile,
  reset: () => {
    hasLoaded = false;
  },
};
