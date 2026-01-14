import type { Message, Source, FollowUpQuestion } from '@/types';
import type { StreamingAction } from './useStreamingChat';

/**
 * Helper functions for useStreamingChat hook
 * Extracted to reduce complexity and improve testability
 */

export interface EventHandlerContext {
  dispatch: React.Dispatch<StreamingAction>;
  pendingMessageRef: React.MutableRefObject<Message | null>;
  flushPendingMessage: () => void;
  setConversationId: (id: string) => void;
  conversationId: string | null;
  messageId: string;
}

export interface TokenEventContext extends EventHandlerContext {
  streamingContent: string;
  streamHasMarkdownSyntax: boolean;
  lastMarkdownCheckAt: number;
  lastPlainFormatAt: number;
  markdownCheckIntervalMs: number;
  plainFormatIntervalMs: number;
  markdownPattern: RegExp;
  formatPlainTextToMarkdown: (text: string) => string | null;
}

/**
 * Handle retrieval_start event
 */
export const handleRetrievalStart = (ctx: EventHandlerContext): void => {
  ctx.dispatch({
    type: 'SET_RETRIEVAL_STATUS',
    status: 'Searching trusted sources...',
  });
};

/**
 * Handle retrieval_complete event
 */
export const handleRetrievalComplete = (ctx: EventHandlerContext): void => {
  ctx.dispatch({
    type: 'SET_RETRIEVAL_STATUS',
    status: 'Preparing answer...',
  });
};

/**
 * Handle sources event
 */
export const handleSourcesEvent = (
  ctx: EventHandlerContext,
  sources: Source[],
  toSources: (rawSources: unknown[]) => Source[],
): void => {
  const processedSources = toSources(sources as unknown[]);
  
  if (ctx.pendingMessageRef.current) {
    ctx.pendingMessageRef.current.sources =
      processedSources.length > 0 ? processedSources : undefined;
    ctx.flushPendingMessage();
  } else if (processedSources.length > 0) {
    ctx.dispatch({
      type: 'SET_MESSAGES',
      updater: (prev) => {
        if (prev.length === 0) return prev;
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        updated[lastIndex] = { ...updated[lastIndex], sources: processedSources };
        return updated;
      },
    });
  }
};

/**
 * Handle token event
 */
export const handleTokenEvent = (
  ctx: TokenEventContext,
  eventContent: string,
): {
  streamingContent: string;
  streamHasMarkdownSyntax: boolean;
  lastMarkdownCheckAt: number;
  lastPlainFormatAt: number;
} => {
  ctx.dispatch({ type: 'SET_RETRIEVAL_STATUS', status: 'Generating answer...' });
  
  const newStreamingContent = ctx.streamingContent + eventContent;
  const now = Date.now();
  let hasMarkdownSyntax = ctx.streamHasMarkdownSyntax;
  let lastMarkdownCheck = ctx.lastMarkdownCheckAt;
  let lastPlainFormat = ctx.lastPlainFormatAt;

  // Check for markdown syntax
  if (!hasMarkdownSyntax) {
    const hintChars = /[`*_#<\n]/;
    if (
      hintChars.test(eventContent) ||
      now - lastMarkdownCheck >= ctx.markdownCheckIntervalMs * 2
    ) {
      hasMarkdownSyntax = ctx.markdownPattern.test(newStreamingContent);
      lastMarkdownCheck = now;
    }
  }

  // Update pending message
  if (ctx.pendingMessageRef.current) {
    if (hasMarkdownSyntax) {
      ctx.pendingMessageRef.current.content = newStreamingContent;
      ctx.pendingMessageRef.current.isFormatted = true;
    } else if (now - lastPlainFormat >= ctx.plainFormatIntervalMs) {
      const formattedContent = ctx.formatPlainTextToMarkdown(newStreamingContent);
      ctx.pendingMessageRef.current.content = formattedContent || newStreamingContent.trim();
      ctx.pendingMessageRef.current.isFormatted = true;
      lastPlainFormat = now;
    } else {
      ctx.pendingMessageRef.current.content = newStreamingContent;
      ctx.pendingMessageRef.current.isFormatted = false;
    }
    ctx.flushPendingMessage();
  }

  return {
    streamingContent: newStreamingContent,
    streamHasMarkdownSyntax: hasMarkdownSyntax,
    lastMarkdownCheckAt: lastMarkdownCheck,
    lastPlainFormatAt: lastPlainFormat,
  };
};

/**
 * Handle metadata event
 */
export const handleMetadataEvent = (
  ctx: EventHandlerContext,
  event: {
    conversation_id?: string;
    follow_up_questions?: unknown[];
  },
  mapFollowUpQuestions: (messageId: string, questions: unknown[]) => FollowUpQuestion[],
): FollowUpQuestion[] => {
  if (event.conversation_id && !ctx.conversationId) {
    ctx.setConversationId(event.conversation_id);
  }

  let followUpQuestions: FollowUpQuestion[] = [];
  
  if (event.follow_up_questions && Array.isArray(event.follow_up_questions)) {
    followUpQuestions = mapFollowUpQuestions(ctx.messageId, event.follow_up_questions);
    
    if (ctx.pendingMessageRef.current) {
      ctx.pendingMessageRef.current.followUpQuestions =
        followUpQuestions.length > 0 ? followUpQuestions : undefined;
      ctx.flushPendingMessage();
    } else if (followUpQuestions.length > 0) {
      ctx.dispatch({
        type: 'SET_MESSAGES',
        updater: (prev) => {
          if (prev.length === 0) return prev;
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          updated[lastIndex] = {
            ...updated[lastIndex],
            followUpQuestions: followUpQuestions,
          };
          return updated;
        },
      });
    }
  }

  return followUpQuestions;
};

/**
 * Create a user message object
 */
export const createUserMessage = (
  messageText: string,
  modelMode: string | undefined,
  shortAnswerMode: boolean,
): Message => {
  return {
    id: Date.now().toString(),
    content: messageText,
    sender: 'user',
    timestamp: Date.now(),
    modelMode,
    shortAnswerMode,
  };
};

/**
 * Create a pending assistant message object
 */
export const createPendingMessage = (
  messageId: string,
  modelMode: string | undefined,
  shortAnswerMode: boolean,
): Message => {
  return {
    id: messageId,
    content: '',
    sender: 'assistant',
    timestamp: Date.now(),
    sources: undefined,
    followUpQuestions: undefined,
    modelMode,
    shortAnswerMode,
  };
};

/**
 * Reset streaming state after completion or error
 */
export const resetStreamingState = (
  dispatch: React.Dispatch<import('./useStreamingChat').StreamingAction>,
  pendingMessageRef: React.MutableRefObject<Message | null>,
  abortControllerRef: React.MutableRefObject<AbortController | null>,
  controller: AbortController,
  setLoading = true,
): void => {
  pendingMessageRef.current = null;
  dispatch({ type: 'SET_PENDING', message: null });
  dispatch({ type: 'SET_RETRIEVAL_STATUS', status: null });
  if (setLoading) {
    dispatch({ type: 'SET_LOADING', value: false });
  }
  if (abortControllerRef.current === controller) {
    abortControllerRef.current = null;
  }
};

/**
 * Create an error message for display
 */
export const createErrorMessage = (
  error: unknown,
  shortAnswerMode: boolean,
): Message => ({
  id: (Date.now() + 2).toString(),
  content: `Sorry, I encountered an error while processing your request. Please try again. Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
  sender: 'assistant',
  timestamp: Date.now(),
  shortAnswerMode,
});

/**
 * Build request body for streaming chat
 */
export const buildStreamingChatRequest = (params: {
  message: string;
  model: string;
  provider: string;
  useRAG: boolean;
  shortAnswerMode: boolean;
  conversationId: string | null;
  modelMode: string | undefined;
  chatHistory: Array<{ role: string; content: string }>;
  reasoningEffort?: string;
  responseVerbosity?: 'low';
}): string => {
  const { reasoningEffort, responseVerbosity, ...rest } = params;
  const smartHints =
    reasoningEffort && responseVerbosity
      ? { reasoningEffort, responseVerbosity }
      : {};

  return JSON.stringify({
    ...rest,
    ...smartHints,
  });
};
