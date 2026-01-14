import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { Message } from '@/types/chat';

interface UseScrollBehaviorOptions {
  scrollAreaRef: RefObject<HTMLDivElement>;
  messages: Message[];
}

export const useScrollBehavior = ({ scrollAreaRef, messages }: UseScrollBehaviorOptions) => {
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [showNewPill, setShowNewPill] = useState(false);
  const suppressPillRef = useRef(false);
  const suppressTimerRef = useRef<number | null>(null);

  const clearTimer = useCallback(() => {
    if (suppressTimerRef.current) {
      window.clearTimeout(suppressTimerRef.current);
      suppressTimerRef.current = null;
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    const viewport = scrollAreaRef.current?.querySelector(
      '[data-radix-scroll-area-viewport]',
    ) as HTMLElement | null;
    if (!viewport) return;

    suppressPillRef.current = true;
    clearTimer();

    const force = () => {
      viewport.scrollTop = viewport.scrollHeight;
    };

    force();
    requestAnimationFrame(force);

    suppressTimerRef.current = window.setTimeout(() => {
      suppressPillRef.current = false;
      suppressTimerRef.current = null;
    }, 400);

    setIsAtBottom(true);
    setShowNewPill(false);
  }, [clearTimer, scrollAreaRef]);

  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector(
      '[data-radix-scroll-area-viewport]',
    ) as HTMLElement | null;
    if (!viewport) return undefined;

    let ticking = false;

    const onScroll = () => {
      if (suppressPillRef.current) return;
      if (ticking) return;
      ticking = true;

      requestAnimationFrame(() => {
        const distance = viewport.scrollHeight - (viewport.scrollTop + viewport.clientHeight);
        const atBottom = distance <= 2;
        setIsAtBottom(atBottom);
        if (atBottom) {
          setShowNewPill(false);
        }
        ticking = false;
      });
    };

    onScroll();
    viewport.addEventListener('scroll', onScroll, { passive: true });

    return () => {
      viewport.removeEventListener('scroll', onScroll as EventListener);
    };
  }, [scrollAreaRef]);

  useEffect(() => {
    if (!isAtBottom && messages.length > 0 && !suppressPillRef.current) {
      setShowNewPill(true);
    }
    if (isAtBottom) {
      setShowNewPill(false);
    }
  }, [isAtBottom, messages]);

  const dismissNewPill = useCallback(() => {
    setShowNewPill(false);
  }, []);

  return {
    isAtBottom,
    showNewPill,
    scrollToBottom,
    dismissNewPill,
  };
};
