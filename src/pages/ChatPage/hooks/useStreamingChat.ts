import { useCallback, useEffect, useReducer, useRef } from 'react';
import type { SetStateAction } from 'react';
import { apiClient, ApiError } from '@/api/client';
import { StorageKeys, getModelDisplayName, HISTORY_WINDOW_BY_MODEL } from '@/constants';
import { getLocalStorageItem } from '@/utils/storage';
import type { Message, Source, FollowUpQuestion } from '@/types';
import { formatPlainTextToMarkdown } from '../utils/formatting';
import { mapFollowUpQuestions } from '../utils/followUps';
import { toSources } from '../utils/sourceFormatting';
import {
  handleRetrievalStart,
  handleRetrievalComplete,
  handleSourcesEvent,
  handleTokenEvent as handleTokenEventHelper,
  handleMetadataEvent,
  createUserMessage,
  createPendingMessage,
  buildStreamingChatRequest,
  resetStreamingState,
  createErrorMessage,
  type EventHandlerContext,
  type TokenEventContext,
} from './streamingChatHelpers';

interface UseStreamingChatOptions {
  conversationId: string | null;
  setConversationId: (id: string) => void;
  setCurrentModel: (model: string) => void;
  DEFAULT_MODEL_ID: string;
  useRAG: boolean;
  shortAnswerMode: boolean;
  modelMode: 'fast' | 'smart';
  maintenanceMode?: boolean;
}

interface StreamingState {
  messages: Message[];
  pendingMessage: Message | null;
  isLoading: boolean;
  retrievalStatus: string | null;
}

type MessagesUpdater = SetStateAction<Message[]>;

export type StreamingAction =
  | { type: 'SET_MESSAGES'; updater: MessagesUpdater }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'SET_PENDING'; message: Message | null }
  | { type: 'SET_LOADING'; value: boolean }
  | { type: 'SET_RETRIEVAL_STATUS'; status: string | null }
  | { type: 'FINALIZE_MESSAGE'; message: Message };

const initialState: StreamingState = {
  messages: [],
  pendingMessage: null,
  isLoading: false,
  retrievalStatus: null,
};

const streamingReducer = (state: StreamingState, action: StreamingAction): StreamingState => {
  switch (action.type) {
    case 'SET_MESSAGES': {
      const nextMessages =
        typeof action.updater === 'function'
          ? (action.updater as (prev: Message[]) => Message[])(state.messages)
          : action.updater;
      return {
        ...state,
        messages: nextMessages,
        pendingMessage: nextMessages.length === 0 ? null : state.pendingMessage,
      };
    }
    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.message],
      };
    case 'SET_PENDING':
      return {
        ...state,
        pendingMessage: action.message,
      };
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.value,
      };
    case 'SET_RETRIEVAL_STATUS':
      return {
        ...state,
        retrievalStatus: action.status,
      };
    case 'FINALIZE_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.message],
        pendingMessage: null,
        isLoading: false,
        retrievalStatus: null,
      };
    default:
      return state;
  }
};

const markdownPattern = /```|\n\s*#|\*\*|\n\s*[-*+]\s|\n\s*\d+[.)]\s|<[^>]+>/;

interface RawSource {
  id?: string;
  reference?: string;
  title?: string;
  url?: string;
  content?: string;
  text?: string;
  section?: string;
  page?: number;
  score?: number;
  source?: string;
  metadata?: Record<string, unknown>;
}

interface StreamingEvent {
  type:
    | 'retrieval_start'
    | 'retrieval_complete'
    | 'sources'
    | 'token'
    | 'metadata'
    | 'complete'
    | 'error';
  content?: string;
  sources?: RawSource[];
  conversation_id?: string;
  follow_up_questions?: Array<string | Record<string, unknown>>;
  delta?: unknown;
  message?: string;
}

export const useStreamingChat = ({
  conversationId,
  setConversationId,
  setCurrentModel,
  DEFAULT_MODEL_ID,
  useRAG,
  shortAnswerMode,
  modelMode,
  maintenanceMode = false,
}: UseStreamingChatOptions) => {
  const [state, dispatch] = useReducer(streamingReducer, initialState);
  const { messages, pendingMessage, isLoading, retrievalStatus } = state;

  const abortControllerRef = useRef<AbortController | null>(null);
  const pendingMessageRef = useRef<Message | null>(null);
  const rafIdRef = useRef<number | null>(null);

  const flushPendingMessage = useCallback(() => {
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
    }
    if (!pendingMessageRef.current) return;

    rafIdRef.current = requestAnimationFrame(() => {
      rafIdRef.current = null;
      if (pendingMessageRef.current) {
        dispatch({ type: 'SET_PENDING', message: { ...pendingMessageRef.current } });
      }
    });
  }, []);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, []);

  const setMessages = useCallback((updater: MessagesUpdater) => {
    dispatch({ type: 'SET_MESSAGES', updater });
  }, []);

  const handleStreamingChat = useCallback(
    async (messageText: string) => {
      if (!messageText || isLoading || maintenanceMode) return;

      const userMessage = createUserMessage(messageText, modelMode, shortAnswerMode);

      dispatch({ type: 'ADD_MESSAGE', message: userMessage });
      const currentInput = messageText;
      dispatch({ type: 'SET_LOADING', value: true });
      dispatch({ type: 'SET_RETRIEVAL_STATUS', status: 'Contacting retrieval service...' });

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      pendingMessageRef.current = null;
      dispatch({ type: 'SET_PENDING', message: null });

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const messageId = (Date.now() + 1).toString();
      let streamingContent = '';
      let sources: Source[] = [];
      let followUpQuestions: FollowUpQuestion[] = [];
      let deltaPayload: import('@/types/policy').DeltaResponse | undefined;
      let streamHasMarkdownSyntax = false;
      let lastMarkdownCheckAt = 0;
      let lastPlainFormatAt = 0;
      const markdownCheckIntervalMs = 120;
      const plainFormatIntervalMs = 160;

      try {
        const isTripPlannerMessage = messageText.startsWith('📋 **Trip Plan Request**');
        const userSelectedModel =
          getLocalStorageItem(StorageKeys.selectedModel) || DEFAULT_MODEL_ID;
        const selectedModel = isTripPlannerMessage ? 'gpt-5-mini' : userSelectedModel;
        const historyLimit =
          HISTORY_WINDOW_BY_MODEL[selectedModel] ?? HISTORY_WINDOW_BY_MODEL.default;
        const storedProvider = getLocalStorageItem(StorageKeys.selectedProvider) || 'openai';
        const selectedProvider = isTripPlannerMessage ? 'openai' : storedProvider;

        if (!isTripPlannerMessage) {
          const displayModel = getModelDisplayName(selectedModel);
          setCurrentModel(displayModel);
        }

        const endpoint = '/api/v2/chat/stream';
        const smartHints =
          selectedModel === 'gpt-5-mini'
            ? { reasoningEffort: 'minimal', responseVerbosity: 'low' as const }
            : {};

        const requestBody = buildStreamingChatRequest({
          message: currentInput,
          model: selectedModel,
          provider: selectedProvider,
          useRAG,
          shortAnswerMode,
          conversationId,
          modelMode,
          chatHistory: messages.slice(-historyLimit).map((msg) => ({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.content,
          })),
          reasoningEffort: smartHints.reasoningEffort,
          responseVerbosity: smartHints.responseVerbosity,
        });

        const response = await apiClient.request(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
          },
          body: requestBody,
          signal: controller.signal,
          parseErrorResponse: false,
        });

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('Response body is not readable');
        }

        let buffer = '';

        pendingMessageRef.current = createPendingMessage(messageId, modelMode, shortAnswerMode);
        dispatch({ type: 'SET_PENDING', message: { ...pendingMessageRef.current } });

        // Create reusable event handler context
        const eventHandlerCtx: EventHandlerContext = {
          dispatch,
          pendingMessageRef,
          flushPendingMessage,
          setConversationId,
          conversationId,
          messageId,
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6).trim();
            if (data === '[DONE]' || data === '') continue;

            try {
              const event = JSON.parse(data) as StreamingEvent;
              switch (event.type) {
                case 'retrieval_start':
                  handleRetrievalStart(eventHandlerCtx);
                  break;
                case 'retrieval_complete':
                  handleRetrievalComplete(eventHandlerCtx);
                  break;
                case 'sources':
                  if (event.sources) {
                    sources = toSources(event.sources);
                    handleSourcesEvent(eventHandlerCtx, sources, toSources);
                  }
                  break;
                case 'token':
                  if (event.content) {
                    const tokenCtx: TokenEventContext = {
                      ...eventHandlerCtx,
                      streamingContent,
                      streamHasMarkdownSyntax,
                      lastMarkdownCheckAt,
                      lastPlainFormatAt,
                      markdownCheckIntervalMs,
                      plainFormatIntervalMs,
                      markdownPattern,
                      formatPlainTextToMarkdown,
                    };
                    const tokenResult = handleTokenEventHelper(tokenCtx, event.content);
                    streamingContent = tokenResult.streamingContent;
                    streamHasMarkdownSyntax = tokenResult.streamHasMarkdownSyntax;
                    lastMarkdownCheckAt = tokenResult.lastMarkdownCheckAt;
                    lastPlainFormatAt = tokenResult.lastPlainFormatAt;
                  }
                  break;
                case 'metadata':
                  followUpQuestions = handleMetadataEvent(eventHandlerCtx, event, mapFollowUpQuestions);
                  if (event.delta) {
                    deltaPayload = event.delta as import('@/types/policy').DeltaResponse;
                    if (pendingMessageRef.current) {
                      pendingMessageRef.current.metadata = {
                        ...pendingMessageRef.current.metadata,
                        delta: deltaPayload,
                      };
                      flushPendingMessage();
                    } else {
                      // Attach to the most recent assistant message
                      dispatch({
                        type: 'SET_MESSAGES',
                        updater: (prev) => {
                          if (prev.length === 0) return prev;
                          const updated = [...prev];
                          for (let i = updated.length - 1; i >= 0; i--) {
                            if (updated[i].sender === 'assistant') {
                              updated[i].metadata = {
                                ...updated[i].metadata,
                                delta: deltaPayload,
                              };
                              break;
                            }
                          }
                          return updated;
                        },
                      });
                    }
                  }
                  break;
                case 'complete': {
                  const trimmedContent = streamingContent.trim();
                  const hasMarkdownSyntax =
                    streamHasMarkdownSyntax || markdownPattern.test(trimmedContent);
                  const formattedContent = hasMarkdownSyntax
                    ? trimmedContent
                    : formatPlainTextToMarkdown(trimmedContent);
                  const finalContent = formattedContent || trimmedContent;

                  if (pendingMessageRef.current) {
                    pendingMessageRef.current.content = finalContent;
                    pendingMessageRef.current.isFormatted = true;
                    flushPendingMessage();
                  }

                  const finalMessage: Message = {
                    id: messageId,
                    content: finalContent,
                    sender: 'assistant',
                    timestamp: Date.now(),
                    isFormatted: true,
                    sources: sources.length > 0 ? sources : undefined,
                    followUpQuestions: followUpQuestions.length > 0 ? followUpQuestions : undefined,
                    modelMode,
                    shortAnswerMode,
                    ...(deltaPayload ? { delta: deltaPayload } : {}),
                  };
                  dispatch({ type: 'FINALIZE_MESSAGE', message: finalMessage });
                  pendingMessageRef.current = null;
                  break;
                }
                case 'error':
                  console.error('Streaming error event:', event);
                  throw new Error(event.message || 'Streaming error occurred');
                default:
                  break;
              }
            } catch (parseError) {
              if (data && data !== '') {
                console.error(
                  'Error parsing SSE event:',
                  parseError,
                  'Data:',
                  data.substring(0, 100),
                );
              }
            }
          }
        }

        resetStreamingState(dispatch, pendingMessageRef, abortControllerRef, controller);
      } catch (error) {
        if (error instanceof ApiError) {
          console.error('Streaming service error response:', {
            status: error.status,
            statusText: error.statusText,
          });
        } else {
          console.error('Error with streaming chat:', error);
        }

        if (error instanceof DOMException && error.name === 'AbortError') {
          resetStreamingState(dispatch, pendingMessageRef, abortControllerRef, controller);
          return;
        }

        resetStreamingState(dispatch, pendingMessageRef, abortControllerRef, controller, false);
        dispatch({ type: 'ADD_MESSAGE', message: createErrorMessage(error, shortAnswerMode) });
        dispatch({ type: 'SET_LOADING', value: false });
      }
    },
    [
      conversationId,
      setConversationId,
      setCurrentModel,
      DEFAULT_MODEL_ID,
      useRAG,
      shortAnswerMode,
      modelMode,
      maintenanceMode,
      messages,
      isLoading,
      flushPendingMessage,
    ],
  );

  return {
    messages,
    setMessages,
    pendingMessage,
    isLoading,
    retrievalStatus,
    handleStreamingChat,
  };
};

export default useStreamingChat;
