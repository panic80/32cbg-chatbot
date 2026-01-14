import { EventEmitter } from 'events';
import { PassThrough, Writable } from 'stream';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { pipeStreamingResponse } from '../streaming.js';

const createMockResponse = () => {
  const chunks = [];
  let statusCode = null;
  let headers = {};
  let ended = false;

  const writable = new Writable({
    write(chunk, _encoding, callback) {
      chunks.push(Buffer.from(chunk).toString());
      callback();
    },
  });

  writable.writeHead = (status, headerMap) => {
    statusCode = status;
    headers = headerMap;
  };

  writable.end = (chunk, encoding, callback) => {
    if (chunk) {
      writable.write(chunk, encoding, () => {});
    }
    ended = true;
    Writable.prototype.end.call(writable, callback);
  };

  writable.getChunks = () => chunks;
  writable.getStatus = () => statusCode;
  writable.getHeaders = () => headers;
  Object.defineProperty(writable, 'writableEnded', {
    get: () => ended,
  });

  return writable;
};

describe('pipeStreamingResponse', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('parses metadata events and logs parse errors', () => {
    const upstream = new PassThrough();
    const req = new EventEmitter();
    const res = createMockResponse();
    const onMetadata = vi.fn();
    const logger = {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    };

    pipeStreamingResponse({
      req,
      res,
      upstream: { data: upstream },
      logger,
      onMetadata,
      heartbeatIntervalMs: 0,
      idleTimeoutMs: null,
    });

    upstream.write('data: {"type":"metadata","conversation_id":"abc"}\n\n');
    upstream.write('data: {invalid json}\n\n');
    upstream.end();

    expect(onMetadata).toHaveBeenCalledWith(expect.objectContaining({ conversation_id: 'abc' }));
    expect(logger.warn).toHaveBeenCalledWith(
      'stream.metadataParseError',
      expect.objectContaining({
        sample: expect.any(String),
      }),
    );
  });

  it('emits heartbeat comments on an interval', () => {
    vi.useFakeTimers();
    const upstream = new PassThrough();
    const req = new EventEmitter();
    const res = createMockResponse();

    pipeStreamingResponse({
      req,
      res,
      upstream: { data: upstream },
      heartbeatIntervalMs: 10,
      idleTimeoutMs: null,
    });

    vi.advanceTimersByTime(25);
    upstream.end();

    const output = res.getChunks().join('');
    expect(output).toMatch(/: heartbeat/);
  });

  it('enforces idle timeout and logs the event', () => {
    vi.useFakeTimers();
    const upstream = new PassThrough();
    const req = new EventEmitter();
    const res = createMockResponse();
    const logger = {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    };

    pipeStreamingResponse({
      req,
      res,
      upstream: { data: upstream },
      logger,
      heartbeatIntervalMs: 0,
      idleTimeoutMs: 20,
    });

    vi.advanceTimersByTime(25);

    expect(logger.error).toHaveBeenCalledWith(
      'stream.idleTimeout',
      expect.objectContaining({
        idleTimeoutMs: 20,
      }),
    );
    expect(res.writableEnded).toBe(true);
  });
});
