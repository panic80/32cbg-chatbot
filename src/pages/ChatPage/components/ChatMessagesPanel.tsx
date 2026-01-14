import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Sparkles, Zap, Loader2, Minimize2 } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { SkeletonChatMessage } from '@/components/ui/skeleton';
import { ChatMessage } from './ChatMessage';
import { EmptyState } from './EmptyState';
import { TypingIndicator } from './TypingIndicator';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { Message } from '@/types/chat';

interface ChatMessagesPanelProps {
  scrollAreaRef: React.RefObject<HTMLDivElement>;
  isInitialLoading: boolean;
  messages: Message[];
  visibleMessages: Message[];
  combinedMessages: Message[];
  startIndex: number;
  canShowMoreMessages: boolean;
  showMoreMessages: () => void;
  collapsedMessages: Set<string>;
  onToggleCollapse: (id: string) => void;
  onCopyMessage: (content: string) => void;
  onRegenerateMessage: (id: string) => void;
  onVoiceAction: () => void;
  currentModel: string;
  modelMode: 'fast' | 'smart';
  shortAnswerMode: boolean;
  isLoading: boolean;
  pendingMessage: Message | null;
  onFollowUpClick: (question: string) => void;
  onSuggestionSelect: (value: string) => void;
  retrievalStatus: string | null;
  inputHeight: number;
  pillMargin: number;
  isAtBottom: boolean;
  showNewPill: boolean;
  scrollToBottom: () => void;
  onModePillClick: () => void;
  onShortAnswerPillClick: () => void;
}

export const ChatMessagesPanel: React.FC<ChatMessagesPanelProps> = ({
  scrollAreaRef,
  isInitialLoading,
  messages,
  visibleMessages,
  combinedMessages,
  startIndex,
  canShowMoreMessages,
  showMoreMessages,
  collapsedMessages,
  onToggleCollapse,
  onCopyMessage,
  onRegenerateMessage,
  onVoiceAction,
  currentModel,
  modelMode,
  shortAnswerMode,
  isLoading,
  pendingMessage,
  onFollowUpClick,
  onSuggestionSelect,
  retrievalStatus,
  inputHeight,
  pillMargin,
  isAtBottom,
  showNewPill,
  scrollToBottom,
  onModePillClick,
  onShortAnswerPillClick,
}) => {
  const typingShortAnswers = pendingMessage?.shortAnswerMode ?? shortAnswerMode;
  const handleModePillKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onModePillClick();
    }
  };
  const handleShortPillKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onShortAnswerPillClick();
    }
  };

  return (
    <>
      <ScrollArea ref={scrollAreaRef} className="flex-1 relative">
        {isInitialLoading ? (
          <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-6 sm:pt-8 pb-20 sm:pb-24 space-y-8">
            {[...Array(3)].map((_, i) => (
              <SkeletonChatMessage key={i} variant={i % 2 === 0 ? 'sent' : 'received'} />
            ))}
          </div>
        ) : messages.length === 0 ? (
          <EmptyState
            onSuggestionClick={(title) => {
              onSuggestionSelect(title);
            }}
          />
        ) : (
          <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-6 sm:pt-8 pb-20 sm:pb-24">
            {canShowMoreMessages && (
              <div className="flex justify-center mb-4">
                <button
                  className="px-3 py-1.5 rounded-full bg-[var(--background-secondary)] text-[var(--text)] text-xs border border-[var(--border)] hover:bg-[var(--background-tertiary)]"
                  onClick={showMoreMessages}
                >
                  Show earlier messages
                </button>
              </div>
            )}
            <AnimatePresence>
              {visibleMessages.map((message, idx) => {
                const messageIndex = startIndex + idx;
                const prev = combinedMessages[messageIndex - 1];
                const showDate =
                  !prev ||
                  new Date(prev.timestamp).toDateString() !==
                    new Date(message.timestamp).toDateString();
                return (
                  <React.Fragment key={message.id}>
                    {showDate && (
                      <div className="flex justify-center my-4">
                        <span className="text-xs px-3 py-1 rounded-full bg-[var(--background-secondary)] text-[var(--text-secondary)]">
                          {new Date(message.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                    )}
                    <ChatMessage
                      message={message}
                      messageIndex={messageIndex}
                      isCollapsed={collapsedMessages.has(message.id)}
                      onToggleCollapse={() => onToggleCollapse(message.id)}
                      onCopy={() => onCopyMessage(message.content)}
                      onRegenerate={() => onRegenerateMessage(message.id)}
                      onVoice={() => onVoiceAction()}
                      currentModel={currentModel}
                      modelMode={message.modelMode || modelMode}
                      shortAnswerMode={
                        typeof message.shortAnswerMode === 'boolean'
                          ? message.shortAnswerMode
                          : shortAnswerMode
                      }
                      isLoading={isLoading}
                      isLatestMessage={messageIndex === combinedMessages.length - 1}
                      onFollowUpClick={onFollowUpClick}
                      onModePillClick={onModePillClick}
                      onShortAnswerPillClick={onShortAnswerPillClick}
                    />
                  </React.Fragment>
                );
              })}
            </AnimatePresence>
            {isLoading && !pendingMessage && (
              <motion.div
                className="mr-4 sm:mr-8 lg:mr-12 mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <motion.div
                  className="flex items-center gap-2 mb-3 ml-12 flex-wrap"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <button
                    type="button"
                    onClick={onModePillClick}
                    onKeyDown={handleModePillKeyDown}
                    className={cn(
                      'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[var(--primary)] transition-shadow',
                      modelMode === 'smart'
                        ? 'bg-blue-500/10 text-blue-500 border border-blue-500/20'
                        : 'bg-green-500/10 text-green-500 border border-green-500/20',
                    )}
                    aria-label="Open settings menu"
                  >
                    {modelMode === 'smart' ? (
                      <>
                        <Sparkles size={12} />
                        <span>Smart Mode</span>
                      </>
                    ) : (
                      <>
                        <Zap size={12} />
                        <span>Fast Mode</span>
                      </>
                    )}
                  </button>
                  {typingShortAnswers && (
                    <button
                      type="button"
                      onClick={onShortAnswerPillClick}
                      onKeyDown={handleShortPillKeyDown}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-500 border border-amber-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-amber-500"
                      aria-label="Open settings menu"
                    >
                      <Minimize2 size={12} />
                      <span>Short answers</span>
                    </button>
                  )}
                  <span className="text-xs text-[var(--text-secondary)]">
                    {modelMode === 'smart'
                      ? 'Detailed answers but slower. Select Fast mode in the menu for quicker responses.'
                      : 'Quick responses. Select Smart mode in the menu for detailed answers.'}
                  </span>
                </motion.div>

                <div className="flex gap-4 justify-start">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  >
                    <Avatar className="h-8 w-8 sm:h-10 sm:w-10 border border-[var(--border)] shadow-lg">
                      <AvatarFallback className="bg-[var(--primary)] text-white font-semibold">
                        P
                      </AvatarFallback>
                    </Avatar>
                  </motion.div>
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.1 }}
                  >
                    <Card className="glass border border-[var(--border)] backdrop-blur-xl">
                      <CardContent className="p-6">
                        <TypingIndicator />
                      </CardContent>
                    </Card>
                  </motion.div>
                </div>
              </motion.div>
            )}
          </div>
        )}
      </ScrollArea>

      <AnimatePresence>
        {retrievalStatus && (
          <motion.div
            key="retrieval-status"
            className="absolute left-0 right-0 flex justify-center pointer-events-none"
            style={{
              bottom: Math.max(Math.round(inputHeight) + pillMargin + 48, 128),
              zIndex: 70,
            }}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
          >
            <div className="pointer-events-auto flex items-center gap-2 rounded-full bg-[var(--card)]/95 border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-secondary)] shadow-md backdrop-blur">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--primary)]" />
              <span>{retrievalStatus}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!isAtBottom && showNewPill && (
        <div
          className="absolute left-0 right-0 flex justify-center pointer-events-none"
          style={{
            bottom: Math.max(Math.round(inputHeight) + pillMargin, 88),
            zIndex: 60,
          }}
        >
          <button
            className="pointer-events-auto px-3 py-1.5 rounded-full bg-[var(--primary)] text-white text-xs shadow-md"
            onClick={scrollToBottom}
          >
            Jump to bottom
          </button>
        </div>
      )}
    </>
  );
};

export default ChatMessagesPanel;
