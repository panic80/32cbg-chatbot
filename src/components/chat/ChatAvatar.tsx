import React from 'react';
import { cn } from '@/lib/utils';

interface ChatAvatarProps {
  variant: 'assistant' | 'user';
  label: string;
  hidden?: boolean;
}

export const ChatAvatar: React.FC<ChatAvatarProps> = ({ variant, label, hidden = false }) => {
  if (hidden) {
    return null;
  }

  return (
    <div className="message-avatar">
      <div
        className={cn('avatar', variant === 'assistant' ? 'assistant-avatar' : 'user-avatar')}
        aria-hidden="true"
      >
        <span>{label}</span>
      </div>
    </div>
  );
};
