/**
 * Security configuration for Express application.
 * Handles Helmet (CSP, HSTS, etc.) and CORS settings.
 */

import type { HelmetOptions } from 'helmet';
import type { CorsOptions } from 'cors';
import type { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';

const isDevelopment = process.env.NODE_ENV === 'development';
const isProduction = process.env.NODE_ENV === 'production';

/**
 * Generate a cryptographically secure nonce for CSP.
 * @returns {string} Base64-encoded nonce
 */
export const generateCspNonce = (): string => {
  return crypto.randomBytes(16).toString('base64');
};

/**
 * Middleware to generate and attach CSP nonce to response locals.
 * The nonce should be used in inline scripts and styles.
 * @returns {Function} Express middleware
 */
export const createCspNonceMiddleware =
  () => (req: Request, res: Response, next: NextFunction) => {
    res.locals.cspNonce = generateCspNonce();
    next();
  };

/**
 * Creates Helmet configuration with enhanced security headers.
 * Uses nonces for script-src to avoid unsafe-inline where possible.
 * @returns {HelmetOptions} Helmet options
 */
export const createHelmetConfig = (): HelmetOptions => ({
  crossOriginEmbedderPolicy: false, // Disable COEP to allow Google Maps API
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: [
        "'self'",
        // Note: unsafe-inline is still needed for React inline styles and Tailwind
        // Consider moving to CSS-in-JS with nonces or external stylesheets in future
        "'unsafe-inline'",
        'https://fonts.googleapis.com',
      ],
      scriptSrc: [
        "'self'",
        // Use nonce for inline scripts - safer than unsafe-inline
        // The nonce is dynamically generated per request via createCspNonceMiddleware
        ((req: Request, res: Response) => `'nonce-${res.locals.cspNonce}'`) as unknown as string,
        // Security: unsafe-* CSP directives are NOT included to improve XSS protection
        'https://fonts.googleapis.com',
        'https://maps.googleapis.com', // Google Maps API
        'https://maps.gstatic.com', // Google Maps static content
      ],
      // Note: scriptSrcAttr removed - inline event handlers should use nonces or be refactored
      fontSrc: ["'self'", 'https://fonts.gstatic.com', 'https://r2cdn.perplexity.ai'],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: [
        "'self'",
        'https://api.openai.com',
        'https://api.anthropic.com',
        'https://generativelanguage.googleapis.com', // Gemini API
        'https://maps.googleapis.com', // Google Maps API
        'https://maps.gstatic.com', // Google Maps static content
        'wss:', // For WebSocket connections if needed
        isDevelopment ? 'http://localhost:*' : '',
      ].filter(Boolean) as string[],
      frameSrc: ["'none'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'none'"],
      childSrc: ["'none'"],
      formAction: ["'self'"],
      upgradeInsecureRequests: isProduction ? [] : null,
      blockAllMixedContent: isProduction ? [] : null,
    },
  },
  hsts: isProduction
    ? {
        maxAge: 31536000, // 1 year
        includeSubDomains: true,
        preload: true,
      }
    : false,
  frameguard: {
    action: 'deny', // Prevent clickjacking
  },
  noSniff: true, // X-Content-Type-Options: nosniff
  xssFilter: true, // X-XSS-Protection: 1; mode=block (legacy but still useful)
  referrerPolicy: {
    policy: 'strict-origin-when-cross-origin',
  },
  permittedCrossDomainPolicies: false,
  dnsPrefetchControl: {
    allow: false,
  },
  ieNoOpen: true,
  originAgentCluster: true,
});

/**
 * Creates CORS configuration with environment-specific settings.
 * @param {Object} options
 * @param {Function} options.logger - Logger function for warnings
 * @returns {CorsOptions} CORS options
 */
export const createCorsConfig = ({
  logger,
}: { logger?: { warn: (msg: string) => void } } = {}): CorsOptions => {
  // Security: Use Set for O(1) origin lookup and exact matching
  // This prevents subdomain bypass attacks (e.g., evil.32cbgg8.com)
  const allowedOriginsSet = getAllowedOrigins();

  return {
    origin: function (origin, callback) {
      // Allow requests with no origin (like mobile apps or curl)
      if (!origin) return callback(null, true);

      // Security: Use Set.has() for exact string matching
      // This is safer than indexOf which could have edge cases
      if (allowedOriginsSet.has(origin)) {
        callback(null, true);
      } else {
        logger?.warn?.(`CORS: Blocked request from origin: ${origin}`);
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
    exposedHeaders: [
      'X-RateLimit-Limit',
      'X-RateLimit-Remaining',
      'X-RateLimit-Reset',
      'X-RateLimit-Burst',
    ],
    maxAge: 86400, // Cache preflight requests for 24 hours
  };
};

/**
 * Get allowed origins set for SSE CORS headers.
 * @returns {Set<string>} Set of allowed origins
 */
export const getAllowedOrigins = (): Set<string> => {
  const origins = isDevelopment
    ? [
        'http://localhost:3000',
        'http://localhost:3001',
        'http://localhost:5173',
        process.env.FRONTEND_URL,
      ].filter(Boolean)
    : ['https://32cbgg8.com', 'https://www.32cbgg8.com', process.env.FRONTEND_URL].filter(Boolean);

  return new Set(origins as string[]);
};

/**
 * Build CORS headers for SSE responses.
 * @param {string} originHeader - The Origin header from the request
 * @returns {Object} CORS headers object
 */
export const buildSseCorsHeaders = (originHeader?: string): Record<string, string> => {
  if (!originHeader) {
    return {};
  }

  const allowedOriginsSet = getAllowedOrigins();

  if (allowedOriginsSet.has(originHeader)) {
    return {
      'Access-Control-Allow-Origin': originHeader,
      'Access-Control-Allow-Credentials': 'true',
      Vary: 'Origin',
    };
  }

  return {};
};

/**
 * Creates additional security headers middleware (Permissions Policy, COOP, etc.)
 * @returns {Function} Express middleware
 */
export const createSecurityHeadersMiddleware =
  () => (req: Request, res: Response, next: NextFunction) => {
    // Permissions Policy (formerly Feature Policy)
    res.setHeader(
      'Permissions-Policy',
      'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()',
    );

    // Security: Prevent cross-domain policy file access
    res.setHeader('X-Permitted-Cross-Domain-Policies', 'none');

    // Security: Prevent content type sniffing (redundant with Helmet but explicit)
    res.setHeader('X-Content-Type-Options', 'nosniff');

    // Additional CORS headers for better security
    if (isProduction) {
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    }

    next();
  };
