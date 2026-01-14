/**
 * Error handling middleware.
 * Provides 404 and global error handlers with helpful responses.
 */

import path from 'path';
import type { Request, Response, NextFunction } from 'express';
import { getLogger } from '../services/logger.js';

const logger = getLogger('middleware:errorHandlers');

/**
 * Creates 404 not found handler.
 * @param {Object} options
 * @param {string} options.distPath - Path to static assets
 * @returns {Function} Express middleware
 */
export const createNotFoundHandler =
  ({ distPath }: { distPath: string | null }) =>
  (req: Request, res: Response) => {
    const requestedUrl = req.url;
    let suggestions: string[] = [];

    // Check if URL might be close to a valid endpoint and suggest alternatives
    if (requestedUrl.includes('gemini') || requestedUrl.includes('chat')) {
      suggestions.push('/api/gemini/generateContent', '/api/v2/chat');
    }

    if (requestedUrl.includes('travel') || requestedUrl.includes('instructions')) {
      suggestions.push('/api/travel-instructions');
    }

    if (requestedUrl.includes('health') || requestedUrl.includes('status')) {
      suggestions.push('/health');
    }

    if (requestedUrl.includes('config') || requestedUrl.includes('settings')) {
      suggestions.push('/api/config');
    }

    // If it looks like an API request, provide JSON response
    if (requestedUrl.startsWith('/api/')) {
      interface ApiErrorResponse {
        error: string;
        message: string;
        timestamp: string;
        suggestions?: string[];
      }
      const response: ApiErrorResponse = {
        error: 'Not Found',
        message: `Cannot ${req.method} ${req.url}`,
        timestamp: new Date().toISOString(),
      };

      // Add suggestions if available
      if (suggestions.length > 0) {
        response.suggestions = suggestions;
        response.message += `. Available endpoints that might help: ${suggestions.join(', ')}`;
      } else {
        // Generic suggestion
        response.message += '. Try /api/config for available endpoints.';
      }

      return res.status(404).json(response);
    }

    // Check if this is a request for a static file
    const staticFileExtensions = [
      '.ico',
      '.png',
      '.jpg',
      '.jpeg',
      '.gif',
      '.svg',
      '.css',
      '.js',
      '.json',
      '.woff',
      '.woff2',
      '.ttf',
      '.eot',
    ];
    const hasStaticExtension = staticFileExtensions.some((ext) =>
      req.path.toLowerCase().endsWith(ext),
    );

    if (hasStaticExtension) {
      // For static files, return proper 404
      return res.status(404).send('File not found');
    }

    // For non-API, non-static requests, serve the React app if available (which will handle its own 404)
    if (distPath) {
      res.sendFile(path.join(distPath, 'index.html'));
    } else {
      // Plain text 404 for non-API requests when no React app is available
      res.status(404).send('404 - Page not found');
    }
  };

/**
 * Creates global error handler.
 * @param {Object} options
 * @param {Object} options.chatLogger - Chat logger
 * @param {boolean} options.loggingEnabled - Whether logging is enabled
 * @returns {Function} Express error middleware
 */
export const createGlobalErrorHandler =
  ({
    chatLogger,
    loggingEnabled,
  }: {
    chatLogger?: import('../services/logger.js').Logger;
    loggingEnabled: boolean;
  }) =>
  (
    err: Error & { statusCode?: number; status?: number; code?: string },
    req: Request,
    res: Response,
    _next: NextFunction,
  ) => {
    const errorId = Date.now().toString(36);
    const errorDetails = {
      id: errorId,
      method: req.method,
      path: req.path,
      query: req.query,
      body: req.body ? JSON.stringify(req.body).substring(0, 1000) : undefined,
      headers: {
        'user-agent': req.headers['user-agent'],
        'content-type': req.headers['content-type'],
      },
      error: {
        message: err.message,
        stack: process.env.NODE_ENV === 'production' ? undefined : err.stack,
        code: err.code,
        statusCode: err.statusCode || err.status,
      },
      timestamp: new Date().toISOString(),
    };

    // Log the error with structured data
    logger.error('Global error handler:', JSON.stringify(errorDetails, null, 2));
    if (chatLogger && loggingEnabled) {
      // Fallback to available logger method
      chatLogger.log(errorDetails);
    }

    // Determine status code
    const statusCode = err.statusCode || err.status || 500;

    // Send appropriate response based on content type
    if (req.path.startsWith('/api/')) {
      res.status(statusCode).json({
        error: statusCode === 500 ? 'Internal Server Error' : err.message,
        message:
          process.env.NODE_ENV === 'production'
            ? 'An unexpected error occurred. Please try again later.'
            : err.message,
        errorId,
        timestamp: new Date().toISOString(),
      });
    } else {
      // For non-API routes, send a simple error page
      res.status(statusCode).send(`
      <html>
        <head><title>Error ${statusCode}</title></head>
        <body>
          <h1>Error ${statusCode}</h1>
          <p>${statusCode === 500 ? 'Internal Server Error' : err.message}</p>
          <p>Error ID: ${errorId}</p>
          <p><a href="/">Go to Homepage</a></p>
        </body>
      </html>
    `);
    }
  };
