import React, { useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { ChevronDown, Sparkles, Zap, Minimize2 } from 'lucide-react';
import { MarkdownRenderer } from '@/components/ui/markdown-renderer';
import { MessageActions } from './MessageActions';
import SuggestionController from '@/components/SuggestionController';
import { SourcesDisplay } from '@/components/SourcesDisplay';
import { DiffPanel } from '@/components/DiffPanel';
import { ClassASection } from '@/components/ClassASection';
import { BlufSection } from '@/components/BlufSection';
import { extractClassASection } from '@/utils/classASection';
import { extractBlufSection } from '@/utils/blufSection';
import type { Message as ChatMessageType } from '@/types/chat';
import { areFollowUpQuestionsEqual } from '../utils/followUps';

const ENABLE_MESSAGE_ACTIONS = false;
const ENABLE_SOURCES_SECTION = true;

interface ChatMessageProps {
  message: ChatMessageType;
  messageIndex: number;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onCopy: () => void;
  onRegenerate: () => void;
  onVoice: () => void;
  currentModel: string;
  modelMode?: 'fast' | 'smart';
  shortAnswerMode: boolean;
  isLoading: boolean;
  isLatestMessage: boolean;
  onFollowUpClick: (question: string) => void;
  onModePillClick: () => void;
  onShortAnswerPillClick: () => void;
}

const ChatMessageInner: React.FC<ChatMessageProps> = ({
  message,
  messageIndex,
  isCollapsed,
  onToggleCollapse,
  onCopy,
  onRegenerate,
  onVoice,
  currentModel,
  modelMode,
  shortAnswerMode,
  isLoading,
  isLatestMessage,
  onFollowUpClick,
  onModePillClick,
  onShortAnswerPillClick,
}) => {
  const prefersReducedMotion = useReducedMotion();
  const [showSources, setShowSources] = useState(false);
  const shouldTruncate = message.content.length > 500;
  const displayContent =
    shouldTruncate && isCollapsed ? message.content.slice(0, 400) + '...' : message.content;
  const isAssistant = message.sender === 'assistant';

  // Extract BLUF and Class A sections for special styling
  const { blufContent, remainingContent: contentAfterBluf } = isAssistant
    ? extractBlufSection(displayContent)
    : { blufContent: null, remainingContent: displayContent };
  const { mainContent, classAContent } = isAssistant
    ? extractClassASection(contentAfterBluf)
    : { mainContent: contentAfterBluf, classAContent: null };
  const isUser = message.sender === 'user';
  const messageShortAnswerMode = message.shortAnswerMode ?? shortAnswerMode;
  const hasSources =
    ENABLE_SOURCES_SECTION &&
    isAssistant &&
    Array.isArray(message.sources) &&
    message.sources.length > 0;
  const sourceList = message.sources ?? [];
  const sourcesSection = hasSources ? (
    <div className="mt-4 pt-4 border-t border-[var(--border)]">
      <button
        className="text-xs px-2 py-1 rounded-full bg-[var(--background-secondary)] hover:bg-[var(--background-tertiary)]"
        onClick={() => setShowSources((s) => !s)}
      >
        {showSources ? 'Hide reference excerpts' : `View reference excerpts (${sourceList.length})`}
      </button>
      {showSources && (
        <div className="mt-3">
          <SourcesDisplay sources={sourceList} />
        </div>
      )}
    </div>
  ) : null;
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
    <motion.div
      key={message.id}
      className={cn(
        'mb-6 sm:mb-8',
        message.sender === 'user' ? 'ml-4 sm:ml-8 lg:ml-12' : 'mr-4 sm:mr-8 lg:mr-12',
      )}
      initial={prefersReducedMotion ? undefined : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
      exit={prefersReducedMotion ? undefined : { opacity: 0, y: -20 }}
      transition={{
        delay: messageIndex * 0.05,
        type: 'spring',
        stiffness: 500,
        damping: 30,
      }}
    >
      <div className={cn('flex gap-2 sm:gap-4', isUser ? 'justify-end' : 'justify-start')}>
        <motion.div
          className={cn('max-w-full sm:max-w-[85%] lg:max-w-[85%] group', isUser ? 'order-1' : '')}
          whileHover={prefersReducedMotion ? undefined : { scale: 1.01 }}
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        >
          {isUser ? (
            <Card
              className={cn(
                'border border-[var(--border)] shadow-lg transition-all duration-300',
                isUser
                  ? 'bg-[var(--primary)] text-white border-transparent hover:shadow-2xl'
                  : 'glass hover:shadow-2xl backdrop-blur-xl',
              )}
            >
              <CardContent className="p-4 sm:p-6 relative overflow-hidden">
                {/* Animated gradient overlay on hover */}
                <motion.div
                  className="absolute inset-0 opacity-0 pointer-events-none"
                  style={{
                    background:
                      message.sender === 'user'
                        ? 'radial-gradient(circle at var(--mouse-x) var(--mouse-y), rgba(255,255,255,0.1) 0%, transparent 60%)'
                        : 'radial-gradient(circle at var(--mouse-x) var(--mouse-y), rgba(var(--primary-rgb),0.1) 0%, transparent 60%)',
                  }}
                  whileHover={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                />

                <div
                  className="leading-relaxed text-[var(--text)] relative z-10 text-lg sm:text-base chat-message-text"
                  style={isUser ? { color: 'white' } : {}}
                >
                  {isAssistant && message.isFormatted ? (
                    <>
                      {blufContent && <BlufSection content={blufContent} />}
                      <MarkdownRenderer>{mainContent}</MarkdownRenderer>
                      {classAContent && <ClassASection content={classAContent} />}
                    </>
                  ) : (
                    <div className="whitespace-pre-wrap break-words overflow-hidden">
                      {displayContent}
                    </div>
                  )}
                  {isAssistant && message.delta && <DiffPanel delta={message.delta} />}

                  {shouldTruncate && (
                    <motion.button
                      className="mt-3 px-3 py-1.5 text-sm font-medium bg-[var(--primary)] text-white rounded-full hover:bg-[var(--primary-hover)] transition-all duration-200 flex items-center gap-1.5 shadow-sm hover:shadow-md"
                      onClick={onToggleCollapse}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {isCollapsed ? 'Read more' : 'Show less'}
                      <ChevronDown
                        size={16}
                        className={cn(
                          'transition-transform duration-200',
                          isCollapsed ? '' : 'rotate-180',
                        )}
                      />
                    </motion.button>
                  )}
                </div>

                {/* Sources */}
                {sourcesSection}
              </CardContent>
            </Card>
          ) : (
            <div>
              {/* Mode Indicator for Assistant Messages */}
              {message.sender === 'assistant' && modelMode && (
                <motion.div
                  className="flex items-center gap-2 mb-3 flex-wrap"
                  initial={prefersReducedMotion ? undefined : { opacity: 0, y: -10 }}
                  animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
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
                  {messageShortAnswerMode && (
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
              )}

              <div
                className="leading-relaxed text-[var(--text)] text-lg sm:text-base chat-message-text"
                role={message.sender === 'assistant' ? 'status' : undefined}
                aria-live={message.sender === 'assistant' ? 'polite' : undefined}
              >
                {message.sender === 'assistant' && message.isFormatted ? (
                  <>
                    {blufContent && <BlufSection content={blufContent} />}
                    <MarkdownRenderer>{mainContent}</MarkdownRenderer>
                    {classAContent && <ClassASection content={classAContent} />}
                  </>
                ) : (
                  <div className="whitespace-pre-wrap break-words overflow-hidden">
                    {displayContent}
                  </div>
                )}

                {shouldTruncate && (
                  <motion.button
                    className="mt-3 px-3 py-1.5 text-sm font-medium bg-[var(--background-secondary)] text-[var(--text)] rounded-full hover:bg-[var(--background-tertiary)] transition-all duration-200 flex items-center gap-1.5 shadow-sm hover:shadow-md"
                    onClick={onToggleCollapse}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {isCollapsed ? 'Read more' : 'Show less'}
                    <ChevronDown
                      size={16}
                      className={cn(
                        'transition-transform duration-200',
                        isCollapsed ? '' : 'rotate-180',
                      )}
                    />
                  </motion.button>
                )}
              </div>

              {sourcesSection}
            </div>
          )}

          {/* Message Actions */}
          {ENABLE_MESSAGE_ACTIONS && message.sender === 'assistant' && (
            <MessageActions
              onCopy={onCopy}
              onRegenerate={onRegenerate}
              onVoice={onVoice}
              isLoading={isLoading}
              onFeedback={(value) => {
                // noop for now; could post to backend
              }}
            />
          )}
        </motion.div>

        {isUser && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: messageIndex * 0.05 + 0.1 }}
          >
            <Avatar className="h-8 w-8 sm:h-10 sm:w-10 border border-[var(--border)] shadow-lg">
              <AvatarFallback className="bg-[var(--primary)] text-white font-semibold">
                U
              </AvatarFallback>
            </Avatar>
          </motion.div>
        )}
      </div>

      <motion.div
        className="text-sm sm:text-xs text-[var(--text-secondary)] mt-2 px-2 flex items-center gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.7 }}
        transition={{ delay: 0.5 }}
      >
        {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        {isAssistant && modelMode && (
          <span className="inline-flex items-center gap-1 ml-2">
            <span className="opacity-60">•</span>
            <span
              className={cn(
                'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium',
                modelMode === 'smart'
                  ? 'bg-blue-500/10 text-blue-500 border border-blue-500/20'
                  : 'bg-green-500/10 text-green-500 border border-green-500/20',
              )}
            >
              {modelMode === 'smart' ? (
                <>
                  <Sparkles size={10} />
                  <span>Smart</span>
                </>
              ) : (
                <>
                  <Zap size={10} />
                  <span>Fast</span>
                </>
              )}
            </span>
            <span className="opacity-60">• {currentModel}</span>
          </span>
        )}
      </motion.div>

      {/* Enhanced Follow-up Questions with Smart Progressive Disclosure */}
      {isAssistant && message.followUpQuestions && message.followUpQuestions.length > 0 && (
        <div className="w-full mt-4">
          <SuggestionController
            questions={message.followUpQuestions}
            onQuestionClick={onFollowUpClick}
            messageId={message.id}
            isLatestMessage={isLatestMessage}
            className=""
          />
        </div>
      )}
    </motion.div>
  );
};

export const ChatMessage = React.memo(ChatMessageInner, (prev, next) => {
  return (
    prev.message.id === next.message.id &&
    prev.message.content === next.message.content &&
    areFollowUpQuestionsEqual(prev.message.followUpQuestions, next.message.followUpQuestions) &&
    prev.isCollapsed === next.isCollapsed &&
    prev.isLoading === next.isLoading &&
    prev.isLatestMessage === next.isLatestMessage &&
    prev.currentModel === next.currentModel &&
    prev.modelMode === next.modelMode &&
    prev.shortAnswerMode === next.shortAnswerMode &&
    prev.onFollowUpClick === next.onFollowUpClick &&
    prev.onModePillClick === next.onModePillClick &&
    prev.onShortAnswerPillClick === next.onShortAnswerPillClick
  );
});
