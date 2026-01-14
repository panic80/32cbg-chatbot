import { useCallback, useMemo, useState } from 'react';
import type { Message } from '@/types/chat';

interface UseMessageWindowOptions {
  messages: Message[];
  pendingMessage: Message | null;
  initialVisible?: number;
  pageSize?: number;
}

export const useMessageWindow = ({
  messages,
  pendingMessage,
  initialVisible = 50,
  pageSize = 50,
}: UseMessageWindowOptions) => {
  const [visibleCount, setVisibleCount] = useState(initialVisible);

  const combinedMessages = useMemo(
    () => (pendingMessage ? [...messages, pendingMessage] : messages),
    [messages, pendingMessage],
  );

  const startIndex = useMemo(
    () => Math.max(0, combinedMessages.length - visibleCount),
    [combinedMessages, visibleCount],
  );

  const visibleMessages = useMemo(
    () => combinedMessages.slice(startIndex),
    [combinedMessages, startIndex],
  );

  const canShowMore = combinedMessages.length > visibleCount;

  const showMore = useCallback(() => {
    setVisibleCount((count) => Math.min(combinedMessages.length, count + pageSize));
  }, [combinedMessages.length, pageSize]);

  const reset = useCallback(() => {
    setVisibleCount(initialVisible);
  }, [initialVisible]);

  return {
    combinedMessages,
    visibleMessages,
    startIndex,
    canShowMore,
    showMore,
    reset,
  };
};

export default useMessageWindow;
