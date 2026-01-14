import { renderHook, act } from '@testing-library/react';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useMobileFlag } from '@/hooks/useMobileFlag';

describe('useMobileFlag', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    document.documentElement.removeAttribute('data-mobile');
    window.innerWidth = 1024;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('updates the root data-mobile attribute when crossing the breakpoint', () => {
    const { result } = renderHook(() => useMobileFlag(768));

    expect(result.current).toBe(false);
    expect(document.documentElement.getAttribute('data-mobile')).toBe('false');

    act(() => {
      window.innerWidth = 600;
      window.dispatchEvent(new Event('resize'));
      vi.advanceTimersByTime(200);
    });

    expect(result.current).toBe(true);
    expect(document.documentElement.getAttribute('data-mobile')).toBe('true');

    act(() => {
      window.innerWidth = 900;
      window.dispatchEvent(new Event('resize'));
      vi.advanceTimersByTime(200);
    });

    expect(result.current).toBe(false);
    expect(document.documentElement.getAttribute('data-mobile')).toBe('false');
  });
});
