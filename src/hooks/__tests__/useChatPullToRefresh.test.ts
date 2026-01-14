import { describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useChatPullToRefresh } from '@/hooks/useChatPullToRefresh';

const createTouchEvent = (clientY: number) =>
  ({
    touches: [{ clientY }],
  }) as unknown as React.TouchEvent;

describe('useChatPullToRefresh', () => {
  it('tracks pull distance and resets on touch end', () => {
    const container = document.createElement('div');
    Object.defineProperty(container, 'scrollTop', { value: 0, writable: true });
    const ref = { current: container };

    const { result } = renderHook(() => useChatPullToRefresh(ref));

    act(() => {
      result.current.handleTouchStart(createTouchEvent(100));
    });

    act(() => {
      result.current.handleTouchMove(createTouchEvent(160));
    });

    expect(result.current.pullOffset).toBeGreaterThan(0);

    act(() => {
      result.current.handleTouchEnd();
    });

    expect(result.current.pullOffset).toBe(0);
  });

  it('invokes trigger callback when threshold exceeded', () => {
    const container = document.createElement('div');
    Object.defineProperty(container, 'scrollTop', { value: 0, writable: true });
    const ref = { current: container };
    const onTrigger = vi.fn();

    const { result } = renderHook(() =>
      useChatPullToRefresh(ref, { triggerOffset: 50, onTrigger }),
    );

    act(() => {
      result.current.handleTouchStart(createTouchEvent(100));
    });

    act(() => {
      result.current.handleTouchMove(createTouchEvent(200));
    });

    expect(onTrigger).toHaveBeenCalledTimes(1);
  });
});
