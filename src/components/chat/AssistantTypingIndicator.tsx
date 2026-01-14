import React from 'react';
import { ChatAvatar } from './ChatAvatar';

interface AssistantTypingIndicatorProps {
  isVisible: boolean;
}

export const AssistantTypingIndicator: React.FC<AssistantTypingIndicatorProps> = ({
  isVisible,
}) => {
  if (!isVisible) {
    return null;
  }

  return (
    <div className="message-wrapper assistant-message">
      <ChatAvatar variant="assistant" label="CF" />
      <div className="assistant-plain-content">
        <div className="typing-indicator">
          <div className="typing-dot animate-typing-dot-bounce" />
          <div
            className="typing-dot animate-typing-dot-bounce"
            style={{ animationDelay: '0.1s' }}
          />
          <div
            className="typing-dot animate-typing-dot-bounce"
            style={{ animationDelay: '0.2s' }}
          />
        </div>
      </div>
    </div>
  );
};
