import { PassThrough, Readable } from 'stream';
import type { Request, Response } from 'express';

interface PipeOptions {
  req: Request;
  res: Response;
  upstream: Readable | { data: Readable };
  corsHeaders?: Record<string, string>;
  logger?: import('./logger.js').Logger | ((level: string, data: unknown) => void);
  onMetadata?: (meta: unknown) => void;
  onToken?: (token: string) => void;
  onComplete?: () => void;
  heartbeatIntervalMs?: number;
  idleTimeoutMs?: number;
  traceId?: string;
}

const resolveLogger = (logger: PipeOptions['logger']) => {
  if (!logger) return null;

  if (typeof logger === 'function') {
    return {
      info: (message: string, meta?: unknown) =>
        logger('info', { message, ...((meta as object) || {}) }),
      warn: (message: string, meta?: unknown) =>
        logger('warn', { message, ...((meta as object) || {}) }),
      error: (message: string, meta?: unknown) =>
        logger('error', { message, ...((meta as object) || {}) }),
    };
  }

  return logger as import('./logger.js').Logger;
};

export const pipeStreamingResponse = ({
  req,
  res,
  upstream,
  corsHeaders = {},
  logger,
  onMetadata,
  onToken,
  onComplete,
  heartbeatIntervalMs = 15000,
  idleTimeoutMs,
  traceId,
}: PipeOptions) => {
  const resolvedLogger = resolveLogger(logger);
  const emit = (level: string, message: string, meta?: unknown) => {
    if (
      resolvedLogger &&
      typeof (resolvedLogger as Record<string, unknown>)[level] === 'function'
    ) {
      ((resolvedLogger as Record<string, unknown>)[level] as Function)(message, {
        traceId,
        ...((meta as object) || {}),
      });
    }
  };

  const passThrough = new PassThrough();
  const upstreamStream: Readable = 'data' in upstream ? upstream.data : upstream;

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    ...corsHeaders,
  });

  passThrough.pipe(res);
  emit('info', 'stream.start', {
    corsApplied: Object.keys(corsHeaders).length > 0,
  });

  const shouldParse = typeof onMetadata === 'function' || typeof onToken === 'function';
  let buffer = '';
  let heartbeatTimer: NodeJS.Timeout | null = null;
  let idleTimer: NodeJS.Timeout | null = null;

  const clearTimers = () => {
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  };

  const sendHeartbeat = () => {
    if (passThrough.writableEnded) return;
    passThrough.write(`: heartbeat ${Date.now()}\n\n`);
    emit('debug', 'stream.heartbeat');
  };

  if (heartbeatIntervalMs && heartbeatIntervalMs > 0) {
    heartbeatTimer = setInterval(sendHeartbeat, heartbeatIntervalMs);
  }

  const scheduleIdleTimeout = () => {
    if (!idleTimeoutMs) return;
    if (idleTimer) {
      clearTimeout(idleTimer);
    }
    idleTimer = setTimeout(() => {
      emit('error', 'stream.idleTimeout', { idleTimeoutMs });
      // Use explicit type cast or optional chaining for destroy
      if ('destroy' in upstreamStream && typeof upstreamStream.destroy === 'function') {
        upstreamStream.destroy(new Error('Stream exceeded idle timeout'));
      }
      passThrough.end();
      if (typeof res.end === 'function' && !res.writableEnded) {
        res.end();
      }
      clearTimers();
    }, idleTimeoutMs);
  };

  scheduleIdleTimeout();

  upstreamStream.on('data', (chunk: Buffer | string) => {
    scheduleIdleTimeout();

    const fragment = chunk.toString();
    passThrough.write(fragment);
    if (!shouldParse) {
      return;
    }

    buffer += fragment;

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (!data || data === '[DONE]') continue;

      try {
        const event = JSON.parse(data);
        if (event.type === 'metadata' && typeof onMetadata === 'function') {
          onMetadata(event);
        }
        if (event.type === 'token' && typeof event.content === 'string' && typeof onToken === 'function') {
          onToken(event.content);
        }
      } catch (error) {
        emit('warn', 'stream.metadataParseError', {
          error,
          sample: data.substring(0, 100),
        });
      }
    }
  });

  upstreamStream.on('end', () => {
    clearTimers();
    passThrough.end();
    emit('info', 'stream.complete');
    onComplete?.();
  });

  upstreamStream.on('error', (error: Error) => {
    clearTimers();
    emit('error', 'stream.upstreamError', { error });
    passThrough.end();
    res.end();
  });

  req.on('close', () => {
    clearTimers();
    emit('warn', 'stream.clientDisconnected');
    // Use explicit type cast or optional chaining for destroy
    if ('destroy' in upstreamStream && typeof upstreamStream.destroy === 'function') {
      upstreamStream.destroy();
    }
    passThrough.end();
  });

  return passThrough;
};
