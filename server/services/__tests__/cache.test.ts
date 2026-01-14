import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CacheService from '../cache.js';

const realSetInterval = setInterval;

describe('CacheService', () => {
  const timers: NodeJS.Timeout[] = [];

  beforeEach(() => {
    process.env.ENABLE_CACHE = 'false';

    vi.restoreAllMocks();

    vi.spyOn(global, 'setInterval').mockImplementation(
      (handler: TimerHandler, timeout?: number) => {
        const timer = realSetInterval(handler, timeout);
        if (typeof (timer as NodeJS.Timeout).unref === 'function') {
          (timer as NodeJS.Timeout).unref();
        }
        timers.push(timer as NodeJS.Timeout);
        return timer as unknown as ReturnType<typeof setInterval>;
      },
    );
  });

  afterEach(async () => {
    timers.forEach((timer) => clearInterval(timer));
    timers.length = 0;
    (setInterval as unknown as { mockRestore?: () => void }).mockRestore?.();
    vi.restoreAllMocks();
  });

  const createCache = () =>
    new CacheService({
      redisEnabled: false,
      enableLogging: false,
      defaultTTL: 200,
      memoryCleanupInterval: 5000,
    });

  it('stores, retrieves, and removes values in memory cache when Redis is disabled', async () => {
    const cache = createCache();

    await cache.set('doc', { title: 'manual', version: 1 });
    expect(await cache.has('doc')).toBe(true);

    const fetched = await cache.get('doc');
    expect(fetched).toEqual({ title: 'manual', version: 1 });

    const deleted = await cache.delete('doc');
    expect(deleted).toBe(true);
    expect(await cache.has('doc')).toBe(false);

    await cache.set('other', { draft: true });
    await cache.clear();
    expect(await cache.get('other')).toBeNull();

    await cache.disconnect();
  });

  it('expires entries based on TTL and updates cache statistics', async () => {
    const cache = createCache();

    const nowSpy = vi.spyOn(Date, 'now');
    nowSpy.mockReturnValue(1_000);
    await cache.set('session', { id: 'abc' }, 200);

    nowSpy.mockReturnValue(1_150);
    const active = await cache.get('session');
    expect(active).toEqual({ id: 'abc' });

    nowSpy.mockReturnValue(1_500);
    const expired = await cache.get('session');
    expect(expired).toBeNull();

    const stats = cache.getStats();
    expect(stats.memory.hits).toBe(1);
    expect(stats.memory.misses).toBe(1);
    expect(stats.combined.hitRate).toBe('50.00');

    const health = cache.getHealth();
    expect(health.status).toBe('healthy');
    expect(health.memory.usage).toBe('0/1000');

    nowSpy.mockRestore();
    await cache.disconnect();
  });
});
