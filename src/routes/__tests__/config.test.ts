import { describe, expect, it } from 'vitest';
import { appRoutes, isLazyRoute } from '../config';

describe('appRoutes configuration', () => {
  it('includes chat route as a lazy route with loader', () => {
    const chatRoute = appRoutes.find((route) => route.path === '/chat');
    expect(chatRoute).toBeDefined();
    expect(isLazyRoute(chatRoute!)).toBe(true);

    if (isLazyRoute(chatRoute!)) {
      expect(typeof chatRoute.loader).toBe('function');
    }
  });

  it('does not add prefetch to non-lazy routes', () => {
    const nonLazyRoutesWithPrefetch = appRoutes.filter(
      (route) => !isLazyRoute(route) && 'prefetch' in route,
    );

    expect(nonLazyRoutesWithPrefetch.length).toBe(0);
  });
});
