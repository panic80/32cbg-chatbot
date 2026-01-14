import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { AnimatedButton } from '@/components/ui/animated-button';
import { Input } from '@/components/ui/input';
import { X, Send, Paperclip } from 'lucide-react';
import { toast } from 'sonner';
import { CATEGORIZED_SUGGESTIONS } from '../constants/suggestions';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleKeyPress: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  handleSendMessage: () => void | Promise<void>;
  isLoading: boolean;
  showInlineCommand: boolean;
  selectedCommandIndex: number;
  setShowInlineCommand: React.Dispatch<React.SetStateAction<boolean>>;
  commands: {
    icon: React.ReactNode;
    label: string;
    command: string;
    description: string;
  }[];
  currentModel: string;
  // Optional attachments controls
  attachments?: { id: string; name: string; size?: number }[];
  onAttachFiles?: (files: File[]) => void;
  onRemoveAttachment?: (id: string) => void;
  maintenanceMode?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  handleInputChange,
  handleKeyPress,
  handleSendMessage,
  isLoading,
  showInlineCommand,
  selectedCommandIndex,
  setShowInlineCommand,
  commands,
  currentModel,
  attachments = [],
  onAttachFiles,
  onRemoveAttachment,
  maintenanceMode = false,
}) => {
  const ENABLE_ATTACHMENTS = false;
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const prefersReducedMotion = useReducedMotion();
  const popularQuestions = useMemo(() => {
    return (
      CATEGORIZED_SUGGESTIONS.find((category) => category.id === 'popular')?.questions.map(
        (question) => question.title,
      ) || []
    );
  }, []);
  const [tickerIndex, setTickerIndex] = useState(0);
  useEffect(() => {
    if (popularQuestions.length <= 1) return;
    const interval = window.setInterval(() => {
      setTickerIndex((current) => (current + 1) % popularQuestions.length);
    }, 3600);
    return () => window.clearInterval(interval);
  }, [popularQuestions.length]);
  const showSuggestionTicker = input.length === 0 && popularQuestions.length > 0;
  const currentSuggestion = showSuggestionTicker ? popularQuestions[tickerIndex] : '';

  return (
    <motion.div
      data-chat-input
      className="border-t border-[var(--border)] glass p-3 sm:p-4"
      initial={prefersReducedMotion ? undefined : { y: 100 }}
      animate={prefersReducedMotion ? undefined : { y: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        paddingBottom: `calc(0.75rem + env(safe-area-inset-bottom))`,
        paddingLeft: `calc(0.75rem + env(safe-area-inset-left))`,
        paddingRight: `calc(0.75rem + env(safe-area-inset-right))`,
        backgroundColor: 'var(--background)',
        // Avoid stacked shadows from nested glass elements
        boxShadow: 'none',
      }}
    >
      <div className="max-w-4xl mx-auto overflow-hidden">
        <div className="relative flex items-end gap-4">
          <div className="flex-1 relative">
            {/* Inline Command Palette */}
            <AnimatePresence>
              {showInlineCommand && (
                <motion.div
                  className="absolute bottom-full mb-2 inset-x-0 max-w-full bg-[var(--card)] border border-[var(--border)] rounded-2xl shadow-2xl overflow-hidden backdrop-blur-xl"
                  initial={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
                  animate={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                  exit={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
                >
                  <div className="p-2" role="listbox" aria-label="Inline commands">
                    {commands.map((cmd, index) => (
                      <motion.div
                        key={cmd.command}
                        className={cn(
                          'flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-colors',
                          selectedCommandIndex === index
                            ? 'bg-[var(--primary)] text-white'
                            : 'hover:bg-[var(--background-secondary)]',
                        )}
                        role="option"
                        aria-selected={selectedCommandIndex === index}
                        onClick={() => {
                          setInput(cmd.command + ' ');
                          setShowInlineCommand(false);
                          inputRef.current?.focus();
                        }}
                        whileHover={prefersReducedMotion ? undefined : { x: 5 }}
                      >
                        <div
                          className={cn(
                            'w-8 h-8 rounded-lg flex items-center justify-center',
                            selectedCommandIndex === index
                              ? 'bg-white/20'
                              : 'bg-[var(--background-secondary)]',
                          )}
                        >
                          {cmd.icon}
                        </div>
                        <div className="flex-1">
                          <div className="font-medium text-sm">{cmd.label}</div>
                          <div className="text-xs opacity-70">{cmd.description}</div>
                        </div>
                        <kbd className="text-xs opacity-50">{cmd.command}</kbd>
                      </motion.div>
                    ))}
                    <div className="px-2 pb-2 text-[10px] text-[var(--text-secondary)]">
                      Enter to accept • Esc to close • Tab cycles
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Attachments row (disabled) */}
            {ENABLE_ATTACHMENTS && attachments.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2">
                {attachments.map((file) => (
                  <div
                    key={file.id}
                    className="inline-flex items-center gap-2 px-2 py-1 rounded-full bg-[var(--background-secondary)] text-xs"
                  >
                    <span className="max-w-[160px] truncate" title={file.name}>
                      {file.name}
                    </span>
                    <button
                      aria-label={`Remove ${file.name}`}
                      className="rounded hover:bg-[var(--background-tertiary)] p-1"
                      onClick={() => onRemoveAttachment?.(file.id)}
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <Input
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyPress}
              placeholder={
                maintenanceMode ? 'Chat is temporarily unavailable...' : 'Ask a question...'
              }
              aria-label="Message input"
              disabled={maintenanceMode}
              className={cn(
                'h-[44px] sm:h-[56px] w-full pr-28 sm:pr-36 pl-4 rounded-3xl border-[var(--border)] bg-[var(--card)] focus:bg-[var(--background-secondary)] transition-all duration-300 text-[16px] sm:text-lg text-[var(--text)] placeholder:text-[var(--text-secondary)]',
                showSuggestionTicker && 'placeholder:text-transparent',
                maintenanceMode && 'opacity-50 cursor-not-allowed',
              )}
            />
            {showSuggestionTicker && (
              <div className="pointer-events-none absolute inset-0 flex items-center pl-4 pr-28 sm:pr-36 text-sm sm:text-base text-[var(--text-secondary)]/50">
                <div className="overflow-hidden h-[1.5em] sm:h-[1.75em]">
                  <AnimatePresence mode="popLayout">
                    <motion.span
                      key={currentSuggestion}
                      className="block whitespace-nowrap overflow-hidden text-ellipsis"
                      initial={prefersReducedMotion ? { opacity: 0 } : { y: '100%', opacity: 0 }}
                      animate={prefersReducedMotion ? { opacity: 1 } : { y: '0%', opacity: 1 }}
                      exit={prefersReducedMotion ? { opacity: 0 } : { y: '-100%', opacity: 0 }}
                      transition={{
                        duration: prefersReducedMotion ? 0.2 : 0.45,
                        ease: prefersReducedMotion ? 'linear' : 'easeOut',
                      }}
                    >
                      {`Try asking: ${currentSuggestion}`}
                    </motion.span>
                  </AnimatePresence>
                </div>
              </div>
            )}

            {/* Trailing controls */}
            {/* Vertically center trailing controls within the input field */}
            <div className="absolute right-2 inset-y-0 flex items-center gap-2">
              <AnimatePresence>
                {input.length > 0 && (
                  <motion.div
                    initial={prefersReducedMotion ? undefined : { scale: 0, opacity: 0 }}
                    animate={prefersReducedMotion ? undefined : { scale: 1, opacity: 1 }}
                    exit={prefersReducedMotion ? undefined : { scale: 0, opacity: 0 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  >
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <AnimatedButton
                          size="icon"
                          variant="ghost"
                          className="h-7 w-7 sm:h-8 sm:w-8 rounded-xl"
                          onClick={() => setInput('')}
                          ripple
                          aria-label="Clear input"
                        >
                          <X size={14} />
                        </AnimatedButton>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Clear</p>
                      </TooltipContent>
                    </Tooltip>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Attach button and input (disabled) */}
              {ENABLE_ATTACHMENTS && (
                <>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <AnimatedButton
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 sm:h-8 sm:w-8 rounded-xl"
                        onClick={() => fileInputRef.current?.click()}
                        ripple
                        aria-label="Attach files"
                      >
                        <Paperclip size={14} />
                      </AnimatedButton>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Attach PDF or link</p>
                    </TooltipContent>
                  </Tooltip>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.txt,.md,.json,application/pdf"
                    className="hidden"
                    onChange={(e) => {
                      const files = Array.from(e.target.files || []);
                      if (files.length === 0) return;
                      onAttachFiles?.(files);
                      toast.info(`${files.length} file${files.length > 1 ? 's' : ''} selected`);
                      e.currentTarget.value = '';
                    }}
                  />
                </>
              )}

              <AnimatedButton
                onClick={() => handleSendMessage()}
                disabled={!input.trim() || isLoading || maintenanceMode}
                size="icon"
                className="h-10 w-10 sm:h-10 sm:w-10 rounded-2xl shadow-lg min-w-[44px] min-h-[44px]"
                variant="default"
                ripple
                glow
                aria-label="Send message"
              >
                <motion.div
                  animate={isLoading ? { rotate: 360 } : {}}
                  transition={{ duration: 1, repeat: isLoading ? Infinity : 0, ease: 'linear' }}
                >
                  <Send size={16} />
                </motion.div>
              </AnimatedButton>
            </div>
          </div>
        </div>
        <motion.div
          className="text-[10px] sm:text-xs text-[var(--text-secondary)] text-center mt-1 sm:mt-2 flex items-center justify-center gap-1 sm:gap-2"
          initial={prefersReducedMotion ? undefined : { opacity: 0 }}
          animate={prefersReducedMotion ? undefined : { opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <span className="opacity-70 sm:opacity-100">AI makes mistakes. Verify with FSA.</span>
          <span className="text-[var(--text-secondary)]/60 hidden sm:inline">•</span>
          <span className="text-[var(--text-secondary)]/70 hidden sm:inline">
            Powered by {currentModel}, LangChain and LangGraph
          </span>
        </motion.div>
      </div>
    </motion.div>
  );
};
