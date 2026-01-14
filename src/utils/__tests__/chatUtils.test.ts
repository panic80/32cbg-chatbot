import { describe, expect, it, beforeEach, afterEach } from 'vitest';
import { hasScrolledUp, isMobileDevice, isScrolledToBottom } from '@/utils/chatUtils';
import { MOBILE_BREAKPOINT_PX, SCROLL_BOTTOM_THRESHOLD_PX } from '@/constants';

describe('chatUtils', () => {
  const originalInnerWidth = window.innerWidth;

  beforeEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: originalInnerWidth,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: originalInnerWidth,
    });
  });

  it('detects mobile device based on configured breakpoint', () => {
    Object.defineProperty(window, 'innerWidth', { value: MOBILE_BREAKPOINT_PX - 10 });
    expect(isMobileDevice()).toBe(true);

    Object.defineProperty(window, 'innerWidth', { value: MOBILE_BREAKPOINT_PX + 1 });
    expect(isMobileDevice()).toBe(false);
  });

  it('determines when the user has scrolled up past the threshold', () => {
    const container = document.createElement('div');

    Object.defineProperty(container, 'scrollHeight', { value: 1000, configurable: true });
    Object.defineProperty(container, 'clientHeight', { value: 400, configurable: true });

    Object.defineProperty(container, 'scrollTop', {
      value: 1000 - 400 - (SCROLL_BOTTOM_THRESHOLD_PX + 5),
      configurable: true,
      writable: true,
    });

    expect(hasScrolledUp(container)).toBe(true);

    Object.defineProperty(container, 'scrollTop', {
      value: 1000 - 400 - SCROLL_BOTTOM_THRESHOLD_PX,
      configurable: true,
    });

    expect(hasScrolledUp(container)).toBe(false);
  });

  it('detects when the user is near the bottom of the scroll container', () => {
    const container = document.createElement('div');

    Object.defineProperty(container, 'scrollHeight', { value: 2000, configurable: true });
    Object.defineProperty(container, 'clientHeight', { value: 800, configurable: true });

    Object.defineProperty(container, 'scrollTop', {
      value: 2000 - 800 - (SCROLL_BOTTOM_THRESHOLD_PX - 2),
      configurable: true,
    });

    expect(isScrolledToBottom(container)).toBe(true);

    Object.defineProperty(container, 'scrollTop', {
      value: 2000 - 800 - (SCROLL_BOTTOM_THRESHOLD_PX + 10),
      configurable: true,
    });

    expect(isScrolledToBottom(container)).toBe(false);
  });
});
