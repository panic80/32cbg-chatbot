import axios from 'axios';
import type { Request, Response } from 'express';
import { RAG_SERVICE_URL, TRIP_PLANNER_PREFIX, TRIP_PLANNER_MODEL } from '../config/constants.js';
import { inferJurisdiction, buildTripPlannerHints } from '../services/tripPlannerService.js';
import { processTripPlannerMessage, sendInternalServerError } from '../utils/chatHelpers.js';
import type { Logger } from '../services/logger.js';

export interface StreamingChatConfig {
  chatLogger: Logger;
  getRagAuthHeaders: () => Record<string, string>;
  config: { loggingEnabled?: boolean };
  pipeStreamingResponse: typeof import('../services/streaming.js').pipeStreamingResponse;
  buildSseCorsHeaders: (origin?: string) => Record<string, string>;
  getEnvNumber: (key: string, fallback: number) => number;
  DEFAULT_RAG_STREAM_TIMEOUT_MS: number;
  TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS: string;
}

export const createStreamingChatHandler = ({
  chatLogger,
  getRagAuthHeaders,
  config,
  pipeStreamingResponse,
  buildSseCorsHeaders,
  getEnvNumber,
  DEFAULT_RAG_STREAM_TIMEOUT_MS,
  TRAVEL_PLANNER_ADDITIONAL_INSTRUCTIONS,
}: StreamingChatConfig) => {
  return async (req: Request, res: Response) => {
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

    const { effectiveModel, effectiveProvider, isTripPlanner } = processTripPlannerMessage(
      message,
      model,
      provider,
      {
        prefix: TRIP_PLANNER_PREFIX,
        model: TRIP_PLANNER_MODEL,
        provider: 'openai',
      },
    );

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

      // Aggregation state for logging
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
};
