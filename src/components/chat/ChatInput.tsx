import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, RefreshCw } from 'lucide-react';
import { AnimatedButton } from '@/components/chat/AnimatedButton';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  className?: string;
  placeholder?: string;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  className,
  placeholder = 'Type your message...',
  disabled = false,
}) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + 'px';
    }
  }, [inputValue]);

  const handleSend = useCallback(() => {
    if (inputValue.trim() && !isLoading && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  }, [inputValue, isLoading, disabled, onSendMessage]);

  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className={cn('input-wrapper', className)}>
      <div className="input-field" style={{ touchAction: 'manipulation' }}>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          className="message-input"
          disabled={isLoading || disabled}
          rows={1}
          style={{ fontSize: '16px' }}
          aria-label="Chat message input"
        />

        <AnimatedButton
          onClick={handleSend}
          disabled={!inputValue.trim() || isLoading || disabled}
          title="Send message"
          variant="default"
          size="sm"
          className="send-button"
        >
          {isLoading ? (
            <RefreshCw className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </AnimatedButton>
      </div>
    </div>
  );
};
