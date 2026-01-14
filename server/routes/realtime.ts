import { Router, Request, Response } from 'express';
import { getLogger } from '../services/logger.js';
import { respondWithError } from '../utils/http.js';

const REALTIME_SESSION_ENDPOINT = 'https://api.openai.com/v1/realtime/sessions';
const buildRealtimeConnectEndpoint = (model: string) =>
  `https://api.openai.com/v1/realtime?model=${encodeURIComponent(model)}`;

const isValidApiKey = (key: string | undefined): boolean => {
  if (!key || typeof key !== 'string') {
    return false;
  }
  if (key.includes('your-') || key.includes('-key-here')) {
    return false;
  }
  return key.trim().length > 20;
};

const resolveOpenAiApiKey = (): string | null => {
  const candidates = [
    process.env.OPENAI_API_KEY,
    process.env.RAG_OPENAI_API_KEY,
    process.env.RAG_SERVICE_OPENAI_API_KEY,
  ];
  return candidates.find(isValidApiKey) || null;
};

interface RealtimeRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  chatLogger: import('../services/logger.js').Logger | null;
}

const createRealtimeRoutes = ({ rateLimiter, chatLogger }: RealtimeRoutesConfig) => {
  const router = Router();
  const logger = getLogger('routes:realtime');

  router.post('/api/v2/realtime/session', rateLimiter, async (req: Request, res: Response) => {
    const apiKey = resolveOpenAiApiKey();

    if (!apiKey) {
      return respondWithError(res, {
        status: 500,
        error: 'ConfigurationError',
        message: 'OpenAI API key is not configured on the server.',
        logger,
        cause: new Error('Missing OpenAI API key'),
      });
    }

    try {
      const model = req.body?.model || 'gpt-realtime-mini';
      const voice = req.body?.voice || 'verse';
      const inputAudioFormat = req.body?.inputAudioFormat || 'pcm16';
      const outputAudioFormat = req.body?.outputAudioFormat || 'pcm16';

      const sessionResponse = await fetch(REALTIME_SESSION_ENDPOINT, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'OpenAI-Beta': 'realtime=v1',
        },
        body: JSON.stringify({
          model,
          voice,
          input_audio_format: inputAudioFormat,
          output_audio_format: outputAudioFormat,
        }),
      });

      if (!sessionResponse.ok) {
        const errorText = await sessionResponse.text();
        if (chatLogger) {
          chatLogger.log({
            type: 'realtime-session-error',
            status: sessionResponse.status,
            message: errorText,
            timestamp: new Date().toISOString(),
          });
        }

        return respondWithError(res, {
          status: sessionResponse.status || 502,
          error: 'RealtimeUpstreamError',
          message: 'Failed to create realtime session.',
          logger,
          cause: new Error(errorText),
          details: { model, status: sessionResponse.status },
        });
      }

      const sessionData = await sessionResponse.json();
      return res.json(sessionData);
    } catch (error: unknown) {
      const err = error as Error;
      if (chatLogger) {
        chatLogger.log({
          type: 'realtime-session-error',
          message: err.message,
          stack: err.stack,
          timestamp: new Date().toISOString(),
        });
      }
      return respondWithError(res, {
        status: 500,
        error: 'RealtimeSessionFailed',
        message: 'Unable to create realtime session.',
        logger,
        cause: err,
      });
    }
  });

  router.post('/api/v2/realtime/answer', rateLimiter, async (req: Request, res: Response) => {
    const { clientSecret, sdp, model = 'gpt-realtime-mini' } = req.body || {};

    if (!clientSecret || !sdp) {
      return respondWithError(res, {
        status: 400,
        error: 'BadRequest',
        message: 'Missing client secret or SDP offer.',
        logger,
        level: 'warn',
        details: { hasClientSecret: !!clientSecret, hasSdp: !!sdp },
      });
    }

    try {
      logger.info('Exchanging realtime SDP with OpenAI', { model });
      const answerResponse = await fetch(buildRealtimeConnectEndpoint(model), {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${clientSecret}`,
          'Content-Type': 'application/sdp',
          'OpenAI-Beta': 'realtime=v1',
        },
        body: sdp,
      });

      if (!answerResponse.ok) {
        const errorText = await answerResponse.text();
        if (chatLogger) {
          chatLogger.log({
            type: 'realtime-answer-error',
            status: answerResponse.status,
            timestamp: new Date().toISOString(),
            message: 'Failed to exchange realtime SDP.',
            details: errorText,
          });
        }

        return respondWithError(res, {
          status: answerResponse.status || 502,
          error: 'RealtimeUpstreamError',
          message: 'Failed to exchange realtime SDP.',
          logger,
          cause: new Error(errorText),
          details: { model, status: answerResponse.status },
        });
      }

      const answer = await answerResponse.text();
      logger.info('Realtime SDP exchange succeeded', { model });
      res.setHeader('Content-Type', 'application/sdp');
      return res.send(answer);
    } catch (error: unknown) {
      const err = error as Error;
      if (chatLogger) {
        chatLogger.log({
          type: 'realtime-answer-error',
          message: err.message,
          stack: err.stack,
          timestamp: new Date().toISOString(),
        });
      }

      return respondWithError(res, {
        status: 500,
        error: 'RealtimeAnswerFailed',
        message: 'Unable to exchange realtime SDP.',
        logger,
        cause: err,
      });
    }
  });

  return router;
};

export default createRealtimeRoutes;
