/**
 * System routes for health, config, and deployment info.
 */

import { Router, Request, Response } from 'express';
import axios from 'axios';
import { existsSync, readFileSync } from 'fs';
import path from 'path';
import { RAG_SERVICE_URL } from '../config/constants.js';
import { getLogger } from '../services/logger.js';
import { AiClients } from '../services/aiClients.js';

const logger = getLogger('routes:system');

interface SystemRoutesConfig {
  config: ReturnType<typeof import('../config/index.js').createGatewayConfig>;
  cache: import('../services/cache.js').CacheService | null;
  requireAdminAuth: import('express').RequestHandler;
  chatLogger: import('../services/logger.js').Logger | null;
  aiClients: AiClients;
  apiRequestCounts: Map<string, number> | null;
}

/**
 * Create system routes.
 * @param {Object} options
 * @param {Object} options.config - Gateway configuration
 * @param {Object} options.cache - Cache service
 * @param {Function} options.requireAdminAuth - Admin auth middleware
 * @param {Object} options.chatLogger - Chat logger
 * @param {Object} options.aiClients - AI client instances
 * @param {Object} options.apiRequestCounts - Rate limit request counts
 * @returns {Router} Express router
 */
const createSystemRoutes = ({
  config,
  cache,
  requireAdminAuth,
  chatLogger,
  aiClients,
  apiRequestCounts,
}: SystemRoutesConfig) => {
  const router = Router();
  const { geminiClient, openaiClient, anthropicClient, openrouterClient } = aiClients || {};

  // Advanced health check endpoint with detailed system stats
  router.get('/health', async (req: Request, res: Response) => {
    // Get cache stats
    const cacheStats = cache ? cache.getStats() : null;
    const cacheHealth = cache ? cache.getHealth() : { status: 'disabled' };

    // Try to get travel instructions cache info
    interface CachedInstructions {
      timestamp: number;
      content: string;
    }
    const travelInstructionsCache = cache ? await cache.get<CachedInstructions>('travel-instructions') : null;
    const cacheAge =
      travelInstructionsCache && travelInstructionsCache.timestamp
        ? Math.floor((Date.now() - travelInstructionsCache.timestamp) / 1000) + 's'
        : 'not cached';

    // Basic memory usage information
    const memoryUsage = process.memoryUsage();
    const formatMemory = (bytes: number) => `${Math.round(bytes / 1024 / 1024)} MB`;

    // Format uptime
    const uptime = process.uptime();
    let uptimeStr;
    if (uptime < 60) {
      uptimeStr = `${Math.floor(uptime)}s`;
    } else if (uptime < 3600) {
      uptimeStr = `${Math.floor(uptime / 60)}m ${Math.floor(uptime % 60)}s`;
    } else {
      uptimeStr = `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m`;
    }

    // Rate limiting stats
    const activeClients = apiRequestCounts ? apiRequestCounts.size : 0;
    const clientsAtLimit = apiRequestCounts
      ? Array.from(apiRequestCounts.entries()).filter((entry) => {
          const count = entry[1];
          return count >= (config.rateLimitMax as number);
        }).length
      : 0;

    // Check RAG service health
    let ragHealth: Record<string, unknown> = { status: 'unknown' };
    if (process.env.RAG_SERVICE_URL || req.query.checkRag === 'true') {
      try {
        const ragResponse = await axios.get(`${RAG_SERVICE_URL}/api/v1/health`, { timeout: 5000 });
        ragHealth = ragResponse.data;
      } catch (error: unknown) {
        ragHealth = { status: 'unhealthy', error: (error as Error).message };
      }
    }

    // For detailed health checks, add API connectivity test
    const healthData = {
      status: 'healthy',
      version: '1.0.0',
      uptime: uptimeStr,
      memory: {
        rss: formatMemory(memoryUsage.rss),
        heapTotal: formatMemory(memoryUsage.heapTotal),
        heapUsed: formatMemory(memoryUsage.heapUsed),
      },
      cache: config.cacheEnabled
        ? {
            enabled: true,
            status: cacheHealth.status,
            redis: (cacheHealth as any).redis,
            memory: (cacheHealth as any).memory,
            performance: (cacheHealth as any).performance,
            stats: cacheStats
              ? {
                  totalHits: cacheStats.combined.totalHits,
                  totalMisses: cacheStats.combined.totalMisses,
                  hitRate: cacheStats.combined.hitRate,
                }
              : null,
            travelInstructions: {
              cached: !!travelInstructionsCache,
              age: cacheAge,
              size:
                travelInstructionsCache && travelInstructionsCache.content
                  ? `${Math.round(travelInstructionsCache.content.length / 1024)} KB`
                  : '0',
            },
          }
        : { enabled: false },
      rateLimiting: {
        enabled: config.rateLimitEnabled,
        activeClients,
        clientsAtLimit,
        limit: config.rateLimitMax,
        window: `${config.rateLimitWindow / 1000}s`,
      },
      environment: process.env.NODE_ENV || 'production',
      ragService: ragHealth,
      timestamp: new Date().toISOString(),
    };

    const publicHealthData: Record<string, unknown> = { ...healthData };
    delete publicHealthData.memory;
    delete publicHealthData.rateLimiting;

    if (req.query.admin === 'true') {
      return requireAdminAuth(req, res, () => {
        res.json(healthData);
      });
    }

    res.json(publicHealthData);
  });

  // API configuration endpoint with environment-specific settings
  router.get('/api/config', (_req: Request, res: Response) => {
    // Safe configuration that doesn't expose sensitive info
    const responseConfig = {
      version: '1.0.0',
      buildTime: process.env.BUILD_TIMESTAMP || new Date().toISOString(),
      environment: process.env.NODE_ENV || 'production',
      features: {
        aiChat: true,
        travelInstructions: true,
        rateLimit: config.rateLimitMax,
      },
      models: {
        default: 'gpt-4.1',
        providers: {
          google: !!geminiClient,
          openai: !!openaiClient,
          anthropic: !!anthropicClient,
          openrouter: !!openrouterClient,
        },
      },
      caching: {
        enabled: config.cacheEnabled,
        duration: Math.floor((config.cacheTTL as number) / 1000 / 60) + ' minutes',
      },
      // Public-facing URLs and endpoints
      api: {
        base: '/api',
        travelInstructions: '/api/travel-instructions',
        gemini: '/api/gemini/generateContent',
        chat: '/api/v2/chat',
        chatRag: '/api/v2/chat/rag',
        followup: '/api/v2/followup',
        ingest: '/api/v2/ingest',
        ingestCanada: '/api/v2/ingest/canada-ca',
        sources: '/api/v2/sources',
        sourcesStats: '/api/v2/sources/stats',
        health: '/health',
      },
      // RAG service info (security: do not expose internal service URLs)
      rag: {
        enabled: !!process.env.RAG_SERVICE_URL,
        // Note: serviceUrl intentionally omitted to avoid exposing internal infrastructure
      },
      // Client-side configuration
      client: {
        retryEnabled: true,
        maxRetries: config.maxRetries,
        retryDelay: config.retryDelay,
      },
      timestamp: new Date().toISOString(),
    };

    // Return the safe config
    res.json(responseConfig);
  });

  // Deployment verification endpoint (for debugging cache issues)
  router.get('/api/deployment-info', requireAdminAuth, (_req: Request, res: Response) => {
    const buildInfo: Record<string, unknown> = {
      timestamp: new Date().toISOString(),
      buildTime: process.env.BUILD_TIMESTAMP,
      nodeEnv: process.env.NODE_ENV,
      processUptime: Math.floor(process.uptime()),
      memoryUsage: process.memoryUsage(),
      // Try to read package.json version
      version: '1.0.0',
    };

    // Try to get build info from dist directory
    try {
      const packagePath = path.join(process.cwd(), 'package.json');
      if (existsSync(packagePath)) {
        const pkg = JSON.parse(readFileSync(packagePath, 'utf8'));
        buildInfo.version = pkg.version;
      }
    } catch (err: unknown) {
      logger.info('Could not read package.json:', (err as Error).message);
    }

    // Add cache-busting headers
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');

    res.json(buildInfo);
  });

  // Cache-busting endpoint for forcing client refresh
  router.post('/api/clear-cache', (req: Request, res: Response) => {
    // This endpoint helps with cache busting by providing a new timestamp
    const cacheBreaker = Date.now();

    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');

    res.json({
      message: 'Cache busting initiated',
      timestamp: new Date().toISOString(),
      cacheBreaker,
      buildTime: process.env.BUILD_TIMESTAMP,
      instructions: {
        manual: 'Press Ctrl+F5 (or Cmd+Shift+R on Mac) to force reload',
        programmatic: `Add ?v=${cacheBreaker} to URLs to bypass cache`,
      },
    });
  });

  return router;
};

export default createSystemRoutes;
