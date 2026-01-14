/**
 * Static file serving middleware.
 * Handles serving of static assets with proper MIME types and caching.
 */

import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import { existsSync, statSync } from 'fs';
import { getLogger } from '../services/logger.js';

const logger = getLogger('middleware:staticFiles');

/**
 * Find the first existing directory from a list of paths.
 * @param {string[]} paths - List of paths to check
 * @returns {string|null} First existing directory or null
 */
export const findExistingPath = (paths: string[]): string | null => {
  for (const testPath of paths) {
    if (existsSync(testPath)) {
      try {
        const stats = statSync(testPath);
        if (stats.isDirectory()) {
          return testPath;
        }
      } catch (err: unknown) {
        logger.error(`Error checking path ${testPath}:`, (err as Error).message);
      }
    }
  }
  return null;
};

/**
 * Find dist directory.
 * @param {string} dirname - __dirname of the calling module
 * @returns {string|null} Path to dist directory or null
 */
export const findDistPath = (dirname: string): string | null => {
  const possiblePaths = [
    path.join(dirname, '..', 'dist'),
    path.join(dirname, '..', 'public_html'),
    path.join(process.cwd(), 'dist'),
    path.join(process.cwd(), 'public_html'),
  ];
  return findExistingPath(possiblePaths);
};

/**
 * Find landing page directory.
 * @param {string} dirname - __dirname of the calling module
 * @returns {string|null} Path to landing directory or null
 */
export const findLandingPath = (dirname: string): string | null => {
  const possiblePaths = [
    path.join(dirname, '..', 'public_html', 'landing'),
    path.join(dirname, '..', 'dist', 'landing'),
    path.join(process.cwd(), 'public_html', 'landing'),
    path.join(process.cwd(), 'dist', 'landing'),
    path.join(dirname, 'public_html', 'landing'),
    path.join(dirname, 'dist', 'landing'),
  ];
  return findExistingPath(possiblePaths);
};

/**
 * Create static file headers setter.
 * @returns {Function} setHeaders callback for express.static
 */
const createHeadersSetter = () => (res: Response, filePath: string) => {
  if (filePath.endsWith('.ico')) {
    res.setHeader('Content-Type', 'image/x-icon');
    res.setHeader('Cache-Control', 'public, max-age=604800');
  } else if (filePath.endsWith('.svg')) {
    res.setHeader('Content-Type', 'image/svg+xml');
    res.setHeader('Cache-Control', 'public, max-age=604800');
  } else if (filePath.endsWith('.png')) {
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Cache-Control', 'public, max-age=604800');
  } else if (filePath.endsWith('.css')) {
    res.setHeader('Content-Type', 'text/css');
  } else if (filePath.endsWith('.js')) {
    res.setHeader('Content-Type', 'application/javascript');
  } else if (filePath.endsWith('.html')) {
    res.setHeader('Content-Type', 'text/html');
  }
};

/**
 * Create favicon route handler.
 * @param {string} distPath - Path to static assets
 * @returns {Function} Express route handler
 */
export const createFaviconHandler = (distPath: string) => (req: Request, res: Response) => {
  logger.info('Favicon route hit!');
  const faviconPath = path.join(distPath, 'favicon.ico');
  logger.info('Looking for favicon at:', faviconPath);
  if (existsSync(faviconPath)) {
    logger.info('Favicon found, sending file');
    res.setHeader('Content-Type', 'image/x-icon');
    res.setHeader('Cache-Control', 'public, max-age=604800');
    res.sendFile(faviconPath);
  } else {
    logger.info('Favicon not found');
    res.status(404).send('Favicon not found');
  }
};

/**
 * Create static files middleware with auth protection for certain paths.
 * @param {Object} options
 * @param {string} options.distPath - Path to static assets
 * @param {Function} options.requiresConfigAuth - Function to check if path needs auth
 * @param {Function} options.requireAdminAuth - Admin auth middleware
 * @returns {Function} Express middleware
 */
export const createProtectedStaticMiddleware = ({
  distPath,
  requiresConfigAuth,
  requireAdminAuth,
}: {
  distPath: string;
  requiresConfigAuth: (path: string) => boolean;
  requireAdminAuth: (req: Request, res: Response, next: NextFunction) => void;
}) => {
  const staticOptions = { setHeaders: createHeadersSetter() };

  return (req: Request, res: Response, next: NextFunction) => {
    if (requiresConfigAuth(req.path)) {
      return requireAdminAuth(req, res, () => {
        express.static(distPath, staticOptions)(req, res, next);
      });
    }

    return express.static(distPath, staticOptions)(req, res, next);
  };
};

/**
 * Create MIME type middleware for favicon and icons.
 * @returns {Function} Express middleware
 */
export const createMimeTypeMiddleware = () => (req: Request, res: Response, next: NextFunction) => {
  if (req.path === '/favicon.ico') {
    res.setHeader('Content-Type', 'image/x-icon');
    res.setHeader('Cache-Control', 'public, max-age=604800'); // Cache for 7 days
  } else if (req.path === '/favicon.svg') {
    res.setHeader('Content-Type', 'image/svg+xml');
    res.setHeader('Cache-Control', 'public, max-age=604800');
  } else if (req.path.endsWith('.png') && req.path.includes('icon')) {
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Cache-Control', 'public, max-age=604800');
  }
  next();
};

/**
 * Create landing page routes.
 * @param {Object} options
 * @param {Object} options.app - Express app
 * @param {string} options.landingPath - Path to landing page assets
 */
export const setupLandingRoutes = ({
  app,
  landingPath,
}: {
  app: express.Application;
  landingPath: string | null;
}) => {
  if (!landingPath) {
    logger.error('Could not find landing page directory');
    return;
  }

  // Serve landing page files with proper MIME types
  app.use(
    '/landing',
    express.static(landingPath, {
      setHeaders: createHeadersSetter(),
    }),
  );

  // Explicit route for landing page
  app.get('/landing', (req, res) => {
    const indexPath = path.join(landingPath, 'index.html');
    if (existsSync(indexPath)) {
      res.sendFile(indexPath);
    } else {
      res.status(404).send('Landing page not found');
    }
  });
};

/**
 * Create SPA catch-all handler for React routes.
 * @param {Object} options
 * @param {string} options.distPath - Path to static assets
 * @returns {Function} Express route handler
 */
export const createSpaCatchAllHandler =
  ({ distPath }: { distPath: string | null }) =>
  (req: Request, res: Response, next: NextFunction) => {
    // Skip API routes and landing routes
    if (req.path.startsWith('/api/') || req.path.startsWith('/landing')) {
      return next();
    }

    // Serve React app for all other routes
    if (distPath) {
      res.sendFile(path.join(distPath, 'index.html'));
    } else {
      return next(); // Let 404 handler take over
    }
  };
