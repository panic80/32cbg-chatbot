import React from 'react';
import { MarkdownRenderer } from '@/components/ui/markdown-renderer';
import type { Message as ChatMessage } from '@/types/chat';
import { ChatAvatar } from './ChatAvatar';

interface ChatMessageBubbleProps {
  message: ChatMessage;
  assistantId: string;
  showAvatar: boolean;
  formatTimestamp: (timestamp: number) => string;
  onCopy: (content: string) => void;
}

const ChatMessageBubbleComponent: React.FC<ChatMessageBubbleProps> = ({
  message,
  assistantId,
  showAvatar,
  formatTimestamp,
  onCopy,
}) => {
  const isUser = message.sender === 'user';
  const isAssistant = message.sender === assistantId;

  return (
    <div className={`message-wrapper ${isUser ? 'user-message' : 'assistant-message'}`}>
      {!isUser && <ChatAvatar variant="assistant" label="CF" hidden={!showAvatar} />}

      <div
        className={
          isUser
            ? 'message-bubble user-bubble bg-primary text-primary-foreground animate-fade-up'
            : 'assistant-plain-content'
        }
      >
        <div className="message-content">
          {isAssistant && message.isFormatted ? (
            <MarkdownRenderer>{message.content}</MarkdownRenderer>
          ) : (
            message.content
          )}
        </div>

        <div className="message-meta">
          <span className="timestamp">{formatTimestamp(message.timestamp)}</span>
          {message.status && (
            <span className="status-indicator">
              {message.status === 'sending' && '⏳'}
              {message.status === 'sent' && '✓'}
              {message.status === 'delivered' && '✓✓'}
              {message.status === 'error' && '⚠️'}
            </span>
          )}
          <button
            className="copy-button p-1 hover:bg-muted rounded transition-colors"
            onClick={() => onCopy(message.content)}
            title="Copy message"
            type="button"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
            </svg>
          </button>
        </div>
      </div>

      {isUser && <ChatAvatar variant="user" label="You" hidden={!showAvatar} />}
    </div>
  );
};

export const ChatMessageBubble = React.memo(ChatMessageBubbleComponent);
