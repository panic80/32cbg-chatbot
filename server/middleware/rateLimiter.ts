/**
 * Rate limiting middleware with Redis and in-memory fallback.
 */

import type { Request, Response, NextFunction } from 'express';
import { getLogger } from '../services/logger.js';

const logger = getLogger('middleware:rateLimiter');

interface RateLimitConfig {
  rateLimitEnabled?: boolean;
  rateLimitMax?: number;
  rateLimitBurst?: number;
  rateLimitWindow: number;
  loggingEnabled?: boolean;
}

interface Bucket {
  count: number;
  expiresAt: number;
}

/**
 * Creates rate limiting middleware.
 * @param {Object} options
 * @param {Object} options.config - Gateway configuration
 * @param {Object} options.cache - Cache service instance
 * @param {Object} options.chatLogger - Chat logger for rate limit events
 * @returns {Object} { rateLimiter, rateLimitBuckets, apiRequestCounts }
 */
export const createRateLimiter = ({
  config,
  cache,
  chatLogger,
}: {
  config: RateLimitConfig;
  cache: import('../services/cache.js').CacheService | null;
  chatLogger: import('../services/logger.js').Logger | null;
}) => {
  const rateLimitBuckets = config.rateLimitEnabled ? new Map<string, Bucket>() : null;
  const apiRequestCounts = config.rateLimitEnabled ? new Map<string, number>() : null;

  // Cleanup interval to prevent memory leaks
  if (config.rateLimitEnabled) {
    setInterval(() => {
      const now = Date.now();
      
      // Cleanup expired buckets
      if (rateLimitBuckets) {
        for (const [key, bucket] of rateLimitBuckets.entries()) {
          if (bucket.expiresAt <= now) {
            rateLimitBuckets.delete(key);
          }
        }
      }

      // Cap apiRequestCounts size to prevent unbounded growth
      if (apiRequestCounts && apiRequestCounts.size > 10000) {
        logger.info('Clearing apiRequestCounts map (size limit exceeded)');
        apiRequestCounts.clear();
      }
    }, 60000).unref(); // Run every minute, don't hold process open
  }

  const rateLimiter = async (req: Request, res: Response, next: NextFunction) => {
    if (!config.rateLimitEnabled) {
      return next();
    }

    const clientIP = req.ip || req.socket.remoteAddress || 'unknown';
    const now = Date.now();
    const windowMs = config.rateLimitWindow;
    const retryAfterSeconds = Math.ceil(windowMs / 1000);
    const limit = (config.rateLimitMax || 0) + (config.rateLimitBurst || 0);

    // Shared key per window
    const windowStart = Math.floor(now / windowMs) * windowMs;
    const windowResetSec = Math.ceil((windowStart + windowMs) / 1000);

    let count = 0;
    let usedRedis = false;

    try {
      // Prefer Redis-backed counter when cache (Redis) is connected
      if (cache && cache.redisConnected) {
        const key = `rl:${clientIP}:${windowStart}`;
        // Atomic INCR with expiry using Lua script to prevent race condition
        const luaScript = `
          local count = redis.call('INCR', KEYS[1])
          if count == 1 then
            redis.call('PEXPIRE', KEYS[1], ARGV[1])
          end
          return count
        `;
        count = Number(await cache.eval(luaScript, {
          keys: [key],
          arguments: [String(windowMs)],
        }));
        usedRedis = true;
      }
    } catch (error: unknown) {
      // Fall back to memory on Redis error
      const err = error as Error;
      logger.warn('Redis rate limit failed, using memory fallback', { error: err.message });
      usedRedis = false;
    }

    if (!usedRedis && rateLimitBuckets) {
      // In-memory fallback (per-process)
      const bucket = rateLimitBuckets.get(clientIP);
      if (!bucket || bucket.expiresAt <= now) {
        rateLimitBuckets.set(clientIP, { count: 1, expiresAt: now + windowMs });
        count = 1;
      } else {
        bucket.count += 1;
        count = bucket.count;
      }
    }

    // Track for health/debug
    if (apiRequestCounts) {
      const prev = apiRequestCounts.get(clientIP) || 0;
      apiRequestCounts.set(clientIP, Math.max(prev, count));
    }

    // Headers
    res.setHeader('X-RateLimit-Limit', String(config.rateLimitMax));
    res.setHeader('X-RateLimit-Burst', String(config.rateLimitBurst || 0));
    res.setHeader('X-RateLimit-Remaining', String(Math.max(0, limit - count)));
    res.setHeader('X-RateLimit-Reset', String(windowResetSec));

    if (count > limit) {
      if (config.loggingEnabled && chatLogger) {
        chatLogger.log({
          message: 'Rate limit exceeded',
          clientIP,
          path: req.path,
          requestCount: count,
          windowMs,
        });
      }
      res.setHeader('Retry-After', retryAfterSeconds);
      return res.status(429).json({ error: 'Rate limit exceeded', retryAfter: retryAfterSeconds });
    }

    return next();
  };

  return {
    rateLimiter,
    rateLimitBuckets,
    apiRequestCounts,
  };
};
