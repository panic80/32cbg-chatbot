import { render, screen } from '@testing-library/react';
import React from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import ChatPage from '../ChatPage';
import { ThemeProvider } from '@/context/ThemeContext';

vi.mock('react-router-dom', () => ({
  useLocation: () => ({ search: '' }),
}));

vi.mock('../ChatPage/hooks', () => {
  const noop = vi.fn();
  return {
    useChatController: () => ({
      commandPaletteProps: {
        open: false,
        onOpenChange: noop,
        onCommandSelect: noop,
      },
      chatHeaderProps: {
        theme: 'light',
        toggleTheme: noop,
        modelMode: 'fast' as const,
        setModelMode: noop,
        onTripPlanSubmit: noop,
        shortAnswerMode: false,
        setShortAnswerMode: noop,
        onExportMarkdown: noop,
        onClearConversation: noop,
        onInsertExample: noop,
        menuOpen: false,
        setMenuOpen: noop,
        highlightModelMode: false,
        highlightShortAnswers: false,
      },
      messagesPanelProps: {},
      chatInputProps: {},
      helpDialogProps: { open: false, onOpenChange: noop },
      disclaimerProps: { open: false, onOpenChange: noop },
    }),
  };
});

vi.mock('../ChatPage/components/BackgroundEffects', () => ({
  BackgroundEffects: () => <div data-testid="background-effects" />,
}));

vi.mock('../ChatPage/components/ChatHeader', () => ({
  ChatHeader: () => <div data-testid="chat-header">Header</div>,
}));

vi.mock('../ChatPage/components/ChatInput', () => ({
  ChatInput: () => <div data-testid="chat-input">Input</div>,
}));

vi.mock('../ChatPage/components/ChatMessage', () => ({
  ChatMessage: () => <div data-testid="chat-message">Message</div>,
}));

vi.mock('../ChatPage/components/HelpDialog', () => ({
  HelpDialog: () => <div data-testid="help-dialog" />,
}));

vi.mock('../ChatPage/components/EmptyState', () => ({
  EmptyState: () => <div data-testid="empty-state">Ready to chat</div>,
}));

vi.mock('../ChatPage/components/TypingIndicator', () => ({
  TypingIndicator: () => <div data-testid="typing-indicator" />,
}));

vi.mock('../ChatPage/components/ChatCommandPalette', () => ({
  ChatCommandPalette: () => <div data-testid="command-palette" />,
}));

vi.mock('../ChatPage/components/ChatMessagesPanel', () => ({
  ChatMessagesPanel: () => <div data-testid="messages-panel" />,
}));

describe('ChatPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders chat layout scaffolding', () => {
    render(
      <ThemeProvider>
        <ChatPage />
      </ThemeProvider>,
    );

    expect(screen.getByTestId('chat-header')).toBeInTheDocument();
    expect(screen.getByTestId('chat-input')).toBeInTheDocument();
  });
});
