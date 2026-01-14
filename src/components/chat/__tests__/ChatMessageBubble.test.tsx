import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatMessageBubble } from '@/components/chat/ChatMessageBubble';
import type { Message } from '@/types/chat';

const baseMessage: Message = {
  id: 'message-1',
  content: 'Hello there',
  sender: 'user',
  timestamp: 1700000000000,
};

const renderBubble = (
  message: Message,
  props?: Partial<React.ComponentProps<typeof ChatMessageBubble>>,
) => {
  const formatTimestamp = vi.fn().mockReturnValue('12:34');
  const onCopy = vi.fn();

  render(
    <ChatMessageBubble
      message={message}
      assistantId="assistant"
      showAvatar
      formatTimestamp={formatTimestamp}
      onCopy={onCopy}
      {...props}
    />,
  );

  return { formatTimestamp, onCopy };
};

describe('ChatMessageBubble', () => {
  it('renders user message content and timestamp', () => {
    const { formatTimestamp } = renderBubble(baseMessage);

    expect(screen.getByText('Hello there')).toBeInTheDocument();
    expect(screen.getByText('12:34')).toBeInTheDocument();
    expect(formatTimestamp).toHaveBeenCalledWith(baseMessage.timestamp);
  });

  it('invokes copy handler when copy button is clicked', () => {
    const { onCopy } = renderBubble(baseMessage);

    fireEvent.click(screen.getByTitle('Copy message'));
    expect(onCopy).toHaveBeenCalledWith('Hello there');
  });

  it('renders markdown for assistant messages when formatted', () => {
    renderBubble(
      {
        ...baseMessage,
        id: 'message-2',
        sender: 'assistant',
        content: '**Bold** response',
        isFormatted: true,
      },
      { showAvatar: true },
    );

    expect(screen.getByText('Bold', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText('CF')).toBeInTheDocument();
  });
});
