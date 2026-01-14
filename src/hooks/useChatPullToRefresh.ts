import { useCallback, useRef, useState } from 'react';

interface UseChatPullToRefreshOptions {
  maxOffset?: number;
  triggerOffset?: number;
  onTrigger?: () => void;
}

export function useChatPullToRefresh(
  containerRef: React.RefObject<HTMLElement>,
  { maxOffset = 100, triggerOffset = 80, onTrigger }: UseChatPullToRefreshOptions = {},
) {
  const [pullOffset, setPullOffset] = useState(0);
  const [isPulling, setIsPulling] = useState(false);
  const touchStartYRef = useRef(0);
  const hasTriggeredRef = useRef(false);

  const handleTouchStart = useCallback(
    (event: React.TouchEvent) => {
      if (containerRef.current?.scrollTop === 0) {
        touchStartYRef.current = event.touches[0].clientY;
        setIsPulling(true);
        hasTriggeredRef.current = false;
      }
    },
    [containerRef],
  );

  const handleTouchMove = useCallback(
    (event: React.TouchEvent) => {
      if (!isPulling || containerRef.current?.scrollTop !== 0) {
        return;
      }

      const currentY = event.touches[0].clientY;
      const delta = currentY - touchStartYRef.current;

      if (delta > 0) {
        const clampedOffset = Math.min(delta, maxOffset);
        setPullOffset(clampedOffset);

        if (!hasTriggeredRef.current && clampedOffset >= triggerOffset) {
          hasTriggeredRef.current = true;
          onTrigger?.();
        }
      }
    },
    [containerRef, isPulling, maxOffset, triggerOffset, onTrigger],
  );

  const handleTouchEnd = useCallback(() => {
    setIsPulling(false);
    setPullOffset(0);
    hasTriggeredRef.current = false;
  }, []);

  return {
    pullOffset,
    isPulling,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
  };
}
