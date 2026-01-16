import type { Logger } from '../services/logger.js';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogEmitter = (level: LogLevel, message: string, meta?: unknown) => void;

/**
 * Creates a scoped logger with an emit helper function.
 * Provides consistent logging pattern across services and controllers.
 *
 * @param logger - Base logger instance
 * @param scope - Scope identifier for this logger (e.g., 'service:rag', 'controller:chat')
 * @returns Object with scopedLogger and emit helper
 *
 * @example
 * const { scopedLogger, emit } = createScopedLogger(logger, 'controller:ingestion');
 * emit('info', 'ingestion.started', { url: 'https://example.com' });
 */
export const createScopedLogger = (
  logger: Logger | undefined,
  scope: string,
): { scopedLogger: Logger | undefined; emit: LogEmitter } => {
  const scopedLogger = logger?.child ? logger.child({ scope }) : logger;

  const emit: LogEmitter = (level, message, meta?) => {
    const loggerFunc = (scopedLogger as unknown as Record<string, unknown>)?.[level];
    if (typeof loggerFunc === 'function') {
      (loggerFunc as (msg: string, meta?: unknown) => void)(message, meta);
    }
  };

  return { scopedLogger, emit };
};

/**
 * Type-safe wrapper to call a logger method by level name.
 * Useful when you need to pass the level as a parameter.
 *
 * @param logger - Logger instance
 * @param level - Log level ('debug' | 'info' | 'warn' | 'error')
 * @param message - Log message
 * @param meta - Optional metadata object
 */
export const logByLevel = (
  logger: Logger | undefined,
  level: LogLevel,
  message: string,
  meta?: unknown,
): void => {
  if (!logger) return;

  const loggerFunc = (logger as unknown as Record<string, unknown>)[level];
  if (typeof loggerFunc === 'function') {
    (loggerFunc as (msg: string, meta?: unknown) => void)(message, meta);
  }
};
