/**
 * Admin authentication middleware.
 * Provides Basic auth protection for admin routes.
 */

import type { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';
import { getLogger } from '../services/logger.js';

const logger = getLogger('middleware:adminAuth');

/**
 * Timing-safe string comparison to prevent timing attacks.
 * Uses constant-time comparison to avoid leaking information about the password.
 * @param {string} a - First string
 * @param {string} b - Second string
 * @returns {boolean} True if strings are equal
 */
const timingSafeEqual = (a: string, b: string): boolean => {
  // Ensure both strings are the same length by padding with a secret value
  // This prevents length-based timing attacks
  const bufA = Buffer.from(a, 'utf8');
  const bufB = Buffer.from(b, 'utf8');

  // If lengths differ, compare against a dummy buffer to maintain constant time
  if (bufA.length !== bufB.length) {
    // Create a buffer of the same length as bufA to compare against
    const dummyBuf = Buffer.alloc(bufA.length);
    crypto.timingSafeEqual(bufA, dummyBuf);
    return false;
  }

  return crypto.timingSafeEqual(bufA, bufB);
};

// Brute-force protection configuration
const LOCKOUT_THRESHOLD = 5; // Number of failed attempts before lockout
const LOCKOUT_DURATION_MS = 15 * 60 * 1000; // 15 minutes lockout
const CLEANUP_INTERVAL_MS = 60 * 1000; // Cleanup every minute

// Track failed login attempts by IP
interface FailedAttempt {
  count: number;
  firstAttempt: number;
  lockedUntil?: number;
}

const failedAttempts = new Map<string, FailedAttempt>();

/**
 * Clean up expired lockouts and old failed attempts
 */
const cleanupFailedAttempts = (): void => {
  const now = Date.now();
  for (const [ip, attempt] of failedAttempts.entries()) {
    // Remove if lockout has expired
    if (attempt.lockedUntil && attempt.lockedUntil < now) {
      failedAttempts.delete(ip);
      continue;
    }
    // Remove old attempts that haven't resulted in lockout (older than lockout duration)
    if (!attempt.lockedUntil && now - attempt.firstAttempt > LOCKOUT_DURATION_MS) {
      failedAttempts.delete(ip);
    }
  }
};

// Run cleanup periodically
setInterval(cleanupFailedAttempts, CLEANUP_INTERVAL_MS);

/**
 * Record a failed login attempt for an IP
 * @param {string} ip - Client IP address
 * @returns {boolean} True if the IP is now locked out
 */
const recordFailedAttempt = (ip: string): boolean => {
  const now = Date.now();
  const existing = failedAttempts.get(ip);

  if (existing) {
    // If already locked out, keep the lockout
    if (existing.lockedUntil && existing.lockedUntil > now) {
      return true;
    }

    // If lockout expired, reset
    if (existing.lockedUntil && existing.lockedUntil <= now) {
      failedAttempts.set(ip, { count: 1, firstAttempt: now });
      return false;
    }

    // Increment count
    existing.count++;

    // Check if should lock out
    if (existing.count >= LOCKOUT_THRESHOLD) {
      existing.lockedUntil = now + LOCKOUT_DURATION_MS;
      logger.warn(`Admin auth: IP ${ip} locked out after ${existing.count} failed attempts`);
      return true;
    }

    return false;
  }

  // First failed attempt
  failedAttempts.set(ip, { count: 1, firstAttempt: now });
  return false;
};

/**
 * Check if an IP is currently locked out
 * @param {string} ip - Client IP address
 * @returns {{ locked: boolean; remainingMs?: number }} Lockout status
 */
const isLockedOut = (ip: string): { locked: boolean; remainingMs?: number } => {
  const attempt = failedAttempts.get(ip);
  if (!attempt?.lockedUntil) {
    return { locked: false };
  }

  const now = Date.now();
  if (attempt.lockedUntil > now) {
    return { locked: true, remainingMs: attempt.lockedUntil - now };
  }

  return { locked: false };
};

/**
 * Clear failed attempts for an IP (on successful login)
 * @param {string} ip - Client IP address
 */
const clearFailedAttempts = (ip: string): void => {
  failedAttempts.delete(ip);
};

/**
 * Check if a path requires config panel authentication.
 * @param {string} pathname - Request path
 * @returns {boolean} True if auth required
 */
export const requiresConfigAuth = (pathname = '') => {
  return (
    pathname === '/config' ||
    pathname.startsWith('/config/') ||
    pathname === '/chat/config' ||
    pathname.startsWith('/chat/config/') ||
    pathname === '/resources' ||
    pathname.startsWith('/resources/') ||
    pathname === '/landing-test' ||
    pathname.startsWith('/landing-test/')
  );
};

interface AdminAuthConfig {
  admin?: {
    password?: string;
    user?: string;
    apiToken?: string;
  };
}

/**
 * Creates admin authentication middleware.
 * Reads credentials from configuration or environment variables.
 * @param {Object} [config] - Gateway configuration object
 * @returns {Object} { requireAdminAuth, adminAuthEnabled }
 */
export const createAdminAuthMiddleware = (config: AdminAuthConfig) => {
  const adminPassword = config?.admin?.password || process.env.CONFIG_PANEL_PASSWORD;
  const adminUser = config?.admin?.user || process.env.CONFIG_PANEL_USER || 'admin';
  const adminApiToken = config?.admin?.apiToken || process.env.ADMIN_API_TOKEN;

  const adminAuthEnabled = typeof adminPassword === 'string' && adminPassword.length > 0;

  if (!adminAuthEnabled) {
    throw new Error('CONFIG_PANEL_PASSWORD must be set before starting the server.');
  }

  if (!adminApiToken || adminApiToken.trim().length === 0) {
    throw new Error('ADMIN_API_TOKEN must be set before starting the server.');
  }

  const requireAdminAuth = (req: Request, res: Response, next: NextFunction) => {
    if (req.method === 'OPTIONS') {
      return next();
    }

    // Get client IP (supports trust proxy)
    const clientIp = req.ip || req.socket.remoteAddress || 'unknown';

    // Check if IP is locked out due to failed attempts
    const lockoutStatus = isLockedOut(clientIp);
    if (lockoutStatus.locked) {
      const remainingMinutes = Math.ceil((lockoutStatus.remainingMs || 0) / 60000);
      logger.warn(`Admin auth: Blocked request from locked out IP ${clientIp}`);
      return res.status(429).json({
        error: 'TooManyRequests',
        message: `Too many failed login attempts. Please try again in ${remainingMinutes} minutes.`,
        retryAfter: Math.ceil((lockoutStatus.remainingMs || 0) / 1000),
      });
    }

    const authHeader = req.headers.authorization || '';
    const [scheme, encoded] = authHeader.split(' ');

    if (scheme === 'Basic' && encoded) {
      try {
        const decoded = Buffer.from(encoded, 'base64').toString('utf8');
        const separatorIndex = decoded.indexOf(':');
        if (separatorIndex !== -1) {
          const providedUser = decoded.slice(0, separatorIndex);
          const providedPassword = decoded.slice(separatorIndex + 1);

          // Use timing-safe comparison to prevent timing attacks
          const userValid = timingSafeEqual(providedUser, adminUser);
          const passwordValid = timingSafeEqual(providedPassword, adminPassword);

          if (userValid && passwordValid) {
            // Successful login - clear any failed attempts
            clearFailedAttempts(clientIp);
            return next();
          }
        }
      } catch (error) {
        logger.error('Failed to decode admin auth credentials', error);
      }
    }

    // Failed authentication - record the attempt
    const nowLockedOut = recordFailedAttempt(clientIp);

    if (nowLockedOut) {
      const remainingMinutes = Math.ceil(LOCKOUT_DURATION_MS / 60000);
      return res.status(429).json({
        error: 'TooManyRequests',
        message: `Too many failed login attempts. Please try again in ${remainingMinutes} minutes.`,
        retryAfter: Math.ceil(LOCKOUT_DURATION_MS / 1000),
      });
    }

    res.setHeader('WWW-Authenticate', 'Basic realm="Config", charset="UTF-8"');
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Administrator credentials required to access this resource.',
    });
  };

  return { requireAdminAuth, adminAuthEnabled };
};

/**
 * Get RAG service auth headers.
 * @returns {Object} Authorization headers
 */
export const getRagAuthHeaders = (): { Authorization: string } => {
  const adminApiToken = process.env.ADMIN_API_TOKEN;
  return { Authorization: `Bearer ${adminApiToken}` };
};
