import axios from 'axios';
import type { Request, Response } from 'express';
import { RAG_SERVICE_URL, TRIP_PLANNER_MODEL, TRIP_PLANNER_PREFIX } from '../config/constants.js';
import { inferJurisdiction, buildTripPlannerHints } from '../services/tripPlannerService.js';
import { AiClients, buildOpenAIParams } from '../services/aiClients.js';
import {
  sendConfigurationError,
  sendBadRequestError,
  sendInternalServerError,
  sendRateLimitError,
  sendUnsupportedProviderError,
  processTripPlannerMessage,
  validateMessage,
} from '../utils/chatHelpers.js';

interface ChatControllerConfig {
  chatLogger: import('../services/logger.js').Logger;
  getRagAuthHeaders: () => Record<string, string>;
  aiService: AiClients & { buildOpenAIParams: typeof buildOpenAIParams };
  config: { loggingEnabled?: boolean };
  pipeStreamingResponse: typeof import('../services/streaming.js').pipeStreamingResponse;
  buildSseCorsHeaders: (origin?: string) => Record<string, string>;
  getEnvNumber: (key: string, fallback: number) => number;
  DEFAULT_RAG_STREAM_TIMEOUT_MS: number;
  TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS: string;
}

export const createChatController = ({
  chatLogger,
  getRagAuthHeaders,
  aiService,
  config,
  pipeStreamingResponse,
  buildSseCorsHeaders,
  getEnvNumber,
  DEFAULT_RAG_STREAM_TIMEOUT_MS,
  TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS,
}: ChatControllerConfig) => {
  const { geminiClient, openaiClient, anthropicClient } = aiService;

  const handleGeminiGenerateContent = async (req: Request, res: Response) => {
    try {
      const { prompt, model: modelId } = req.body;

      if (!validateMessage(prompt)) {
        return sendBadRequestError(res, 'Prompt is required and must be a non-empty string');
      }

      if (!geminiClient) {
        return sendConfigurationError(res, 'Gemini');
      }

      const model = geminiClient.getGenerativeModel({ model: modelId || 'gemini-2.5-flash' });
      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      return res.json({ response: text });
    } catch (error: unknown) {
      const err = error as Error;
      chatLogger?.error?.('Gemini API error', { error: err });
      return res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
      });
    }
  };

  const handleStandardChat = async (req: Request, res: Response) => {
    const { message, model, provider } = req.body;

    const { effectiveModel, effectiveProvider } = processTripPlannerMessage(message, model, provider, {
      prefix: TRIP_PLANNER_PREFIX,
      model: TRIP_PLANNER_MODEL,
      provider: 'openai',
    });

    if (!effectiveModel) {
      return sendBadRequestError(res, 'Model parameter is required.');
    }

    if (!effectiveProvider) {
      return sendBadRequestError(res, 'Provider parameter is required.');
    }

    try {
      let responseText = '';

      switch (effectiveProvider) {
        case 'google': {
          if (!geminiClient) {
            return sendConfigurationError(res, 'Google');
          }
          responseText = await geminiClient
            .getGenerativeModel({ model: effectiveModel })
            .generateContent(message.trim())
            .then((result) => result.response.text());
          break;
        }
        case 'openai': {
          if (!openaiClient) {
            return sendConfigurationError(res, 'OpenAI');
          }
          responseText = await openaiClient.chat.completions
            .create(buildOpenAIParams(effectiveModel, [{ role: 'user', content: message.trim() }]))
            .then((completion) => completion.choices[0].message.content || '');
          break;
        }
        case 'anthropic': {
          if (!anthropicClient) {
            return sendConfigurationError(res, 'Anthropic');
          }
          responseText = await anthropicClient.messages
            .create({
              model: effectiveModel,
              max_tokens: 4096,
              messages: [{ role: 'user', content: message.trim() }],
            })
            .then((anthropicMessage) => {
              const firstContent = anthropicMessage.content[0];
              if (firstContent && 'text' in firstContent) {
                return firstContent.text as string;
              }
              return '';
            });
          break;
        }
        default:
          return sendUnsupportedProviderError(res, effectiveProvider);
      }

      if (config?.loggingEnabled) {
        const loggedAt = new Date().toISOString();
        chatLogger.logChat?.(req, {
          timestamp: loggedAt,
          question: message.trim(),
          answer: responseText,
          model: effectiveModel,
          provider: effectiveProvider,
          ragEnabled: false,
          metadata: { route: '/api/v2/chat' },
        });
      }

      return res.json({
        response: responseText,
        sources: [],
        conversation_id: null,
        model: effectiveModel,
      });
    } catch (error: unknown) {
      const err = error as Error & { status?: number };
      chatLogger?.error?.('Error processing chat request', { error: err });

      if (config?.loggingEnabled) {
        chatLogger.logChat?.(req, {
          timestamp: new Date().toISOString(),
          question: message.trim(),
          answer: null,
          model: effectiveModel,
          provider: effectiveProvider,
          ragEnabled: false,
          metadata: {
            route: '/api/v2/chat',
            error: err.message || 'Unknown error',
          },
        });
      }

      if (err.status === 429) {
        return sendRateLimitError(res);
      }

      if (err.status === 401) {
        return sendConfigurationError(res, 'Invalid API key for the selected provider');
      }

      return sendInternalServerError(res, 'An error occurred while processing your request.');
    }
  };

  const handleRagChat = async (req: Request, res: Response) => {
    const {
      message,
      model,
      provider,
      chatHistory,
      conversationId,
      useRAG = true,
      audience,
    } = req.body;

    const { effectiveModel, effectiveProvider } = processTripPlannerMessage(message, model, provider, {
      prefix: TRIP_PLANNER_PREFIX,
      model: TRIP_PLANNER_MODEL,
      provider: 'openai',
    });

    try {
      chatLogger?.info?.('Processing RAG chat request', {
        message: message?.substring(0, 50),
        model: effectiveModel,
        provider: effectiveProvider,
        hasHistory: !!chatHistory,
        conversationId,
      });

      const ragResponse = await axios.post(
        `${RAG_SERVICE_URL}/api/v1/chat`,
        {
          message: message.trim(),
          chat_history: chatHistory || [],
          conversation_id: conversationId,
          provider: effectiveProvider || 'openai',
          model: effectiveModel,
          use_rag: useRAG,
          include_sources: true,
          ...(audience ? { audience } : {}),
        },
        {
          timeout: 30000,
          headers: {
            'Content-Type': 'application/json',
            ...getRagAuthHeaders(),
          },
        },
      );

      return res.json(ragResponse.data);
    } catch (error: unknown) {
      const err = error as Error & {
        code?: string;
        response?: { data: unknown; status: number };
        stack?: string;
      };
      chatLogger?.error?.('RAG chat error', {
        message: err.message,
        code: err.code,
        response: err.response?.data,
        status: err.response?.status,
        stack: err.stack,
      });

      if (err.response) {
        return res.status(err.response.status).json(err.response.data);
      }

      return res.status(502).json({
        error: 'RAG Service Unavailable',
        message: 'Upstream retrieval service failed and no fallback is configured.',
      });
    }
  };

  const handleStreamingChat = async (req: Request, res: Response) => {
    const {
      message,
      model,
      provider,
      chatHistory,
      conversationId,
      useRAG = true,
      shortAnswerMode = false,
      useHybridSearch = false,
      reasoningEffort,
      responseVerbosity,
      audience,
      modelMode,
    } = req.body;

    const { effectiveModel, effectiveProvider, isTripPlanner } = processTripPlannerMessage(message, model, provider, {
      prefix: TRIP_PLANNER_PREFIX,
      model: TRIP_PLANNER_MODEL,
      provider: 'openai',
    });

    try {
      chatLogger?.info?.('Processing streaming chat request', {
        message: message?.substring(0, 50),
        model: effectiveModel,
        provider: effectiveProvider,
        hasHistory: !!chatHistory,
        conversationId,
      });

      const jurisdiction = isTripPlanner ? inferJurisdiction(message) : undefined;
      let messageForRetrieval = message.trim();
      if (isTripPlanner) {
        const hints = buildTripPlannerHints(jurisdiction);
        messageForRetrieval = `${messageForRetrieval}\n\nRetrieval focus: ${hints.join(' | ')}`;
      }

      const ragStreamTimeout =
        getEnvNumber?.('RAG_STREAM_TIMEOUT', DEFAULT_RAG_STREAM_TIMEOUT_MS) ||
        DEFAULT_RAG_STREAM_TIMEOUT_MS;
      const upstreamAbortController = new AbortController();

      const response = await axios.post(
        `${RAG_SERVICE_URL}/api/v1/streaming_chat`,
        {
          message: messageForRetrieval,
          chat_history: chatHistory || [],
          conversation_id: conversationId,
          provider: effectiveProvider || 'openai',
          model: effectiveModel,
          use_rag: useRAG,
          include_sources: true,
          short_answer_mode: shortAnswerMode,
          use_hybrid_search: isTripPlanner ? true : useHybridSearch,
          ...(isTripPlanner
            ? { additionalInstructions: TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS }
            : {}),
          ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
          ...(responseVerbosity ? { response_verbosity: responseVerbosity } : {}),
          ...(jurisdiction ? { jurisdiction } : {}),
          ...(audience ? { audience } : {}),
          ...(modelMode ? { mode: modelMode } : {}),
        },
        {
          responseType: 'stream',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            ...getRagAuthHeaders(),
          },
          timeout: ragStreamTimeout,
          signal: upstreamAbortController.signal,
        },
      );

      const streamingCorsHeaders = buildSseCorsHeaders?.(req.headers.origin) || {};
      const shouldLogStream = Boolean(config?.loggingEnabled);
      const maxAggregatedAnswerChars =
        getEnvNumber?.('STREAM_LOG_ANSWER_MAX_CHARS', 20000) ?? 20000;
      let aggregatedAnswer = '';
      let hasReachedAnswerLimit = false;
      let remoteConversationId = conversationId || null;
      let aggregatedSources: unknown[] = [];
      let aggregatedFollowUps: unknown[] = [];
      const streamStart = Date.now();
      const streamLogger = chatLogger?.child
        ? chatLogger.child({
            scope: 'routes:chat:stream',
            conversationId: conversationId || null,
          })
        : null;

      pipeStreamingResponse({
        req,
        res,
        upstream: response,
        corsHeaders: {
          ...streamingCorsHeaders,
          'X-Accel-Buffering': 'no',
        },
        logger: streamLogger || undefined,
        heartbeatIntervalMs: 15000,
        idleTimeoutMs: DEFAULT_RAG_STREAM_TIMEOUT_MS,
        traceId: req.headers['x-request-id'] as string,
        onMetadata: shouldLogStream
          ? (meta: unknown) => {
              const event = meta as {
                conversation_id?: string;
                sources?: unknown[];
                follow_up_questions?: unknown[];
              };
              if (event.conversation_id) {
                remoteConversationId = event.conversation_id;
              }
              if (Array.isArray(event.sources)) {
                aggregatedSources = event.sources;
              }
              if (Array.isArray(event.follow_up_questions)) {
                aggregatedFollowUps = event.follow_up_questions;
              }
            }
          : undefined,
        onToken: shouldLogStream
          ? (token: string) => {
              if (hasReachedAnswerLimit) return;
              const remaining = maxAggregatedAnswerChars - aggregatedAnswer.length;
              if (remaining <= 0) {
                hasReachedAnswerLimit = true;
                return;
              }
              aggregatedAnswer += token.length > remaining ? token.slice(0, remaining) : token;
              if (aggregatedAnswer.length >= maxAggregatedAnswerChars) {
                hasReachedAnswerLimit = true;
              }
            }
          : undefined,
        onComplete: () => {
          if (shouldLogStream) {
            chatLogger.logChat?.(req, {
              timestamp: new Date().toISOString(),
              question: message.trim(),
              answer: aggregatedAnswer,
              model: effectiveModel,
              provider: effectiveProvider,
              ragEnabled: useRAG,
              conversationId: remoteConversationId,
              latencyMs: Date.now() - streamStart,
              metadata: {
                route: '/api/v2/chat/stream',
                sources: aggregatedSources,
                followUpQuestions: aggregatedFollowUps,
                ...(reasoningEffort ? { reasoningEffort } : {}),
                ...(responseVerbosity ? { responseVerbosity } : {}),
              },
            });
          }
        },
      });

      req.on('close', () => {
        upstreamAbortController.abort();
      });
    } catch (error: unknown) {
      const err = error as Error;
      chatLogger?.error?.('Error with streaming chat', { error: err });

      if (config?.loggingEnabled) {
        chatLogger.logChat?.(req, {
          timestamp: new Date().toISOString(),
          question: message.trim(),
          answer: null,
          model: effectiveModel,
          provider: effectiveProvider,
          ragEnabled: useRAG,
          metadata: {
            route: '/api/v2/chat/stream',
            error: err.message || 'Unknown error',
          },
        });
      }

      return sendInternalServerError(res, 'An error occurred while processing your streaming request.');
    }
  };

  return {
    handleGeminiGenerateContent,
    handleStandardChat,
    handleRagChat,
    handleStreamingChat,
  };
};
