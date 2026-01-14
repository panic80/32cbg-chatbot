import { useCallback } from 'react';
import { toast } from 'sonner';
import type { Dispatch, SetStateAction } from 'react';
import type { Message } from '@/types/chat';

interface UseMessageOperationsOptions {
  setMessages: Dispatch<SetStateAction<Message[]>>;
}

export const useMessageOperations = ({ setMessages }: UseMessageOperationsOptions) => {
  const copyMessage = useCallback((content: string) => {
    navigator.clipboard
      .writeText(content)
      .then(() => toast.success('Copied to clipboard'))
      .catch(() => toast.error('Copy failed'));
  }, []);

  const regenerateMessage = useCallback(
    (id: string) => {
      // Simulate regeneration; production version should call API
      setTimeout(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === id
              ? { ...msg, content: 'This is a regenerated response with updated content.' }
              : msg,
          ),
        );
      }, 1500);
    },
    [setMessages],
  );

  return {
    copyMessage,
    regenerateMessage,
  };
};
