import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  DEFAULT_INGEST_TIMEOUT_MS,
  DEFAULT_MAX_RETRIES,
  DEFAULT_RETRY_DELAY_MS,
  RAG_SERVICE_URL,
} from '../config/constants.js';

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

interface RagServiceConfig {
  httpClient?: AxiosInstance;
  getRagAuthHeaders?: () => Record<string, string>;
  config?: Record<string, unknown>;
  logger?: import('./logger.js').Logger;
}

interface IngestOptions {
  url: string;
  content?: string;
  type?: string;
  metadata?: Record<string, unknown>;
  forceRefresh?: boolean;
}

export const createRagService = ({
  httpClient = axios,
  getRagAuthHeaders,
  config = {},
  logger,
}: RagServiceConfig) => {
  const ragServiceUrl = config.ragServiceUrl || RAG_SERVICE_URL;
  const ingestTimeout = config.ingestTimeout ?? DEFAULT_INGEST_TIMEOUT_MS;
  const maxRetries = config.ingestMaxRetries ?? config.maxRetries ?? DEFAULT_MAX_RETRIES;
  const retryDelayMs = config.ingestRetryDelay ?? config.retryDelay ?? DEFAULT_RETRY_DELAY_MS;

  const scopedLogger = logger?.child ? logger.child({ scope: 'service:rag' }) : logger;
    const emit = (level: string, message: string, meta?: unknown) => {
      const loggerFunc = (scopedLogger as unknown as Record<string, unknown>)[level];
      if (typeof loggerFunc === 'function') {
          (loggerFunc as Function)(message, meta);
      }
    };
  
    const normalizeHttpClient = (): AxiosInstance => {
      if (httpClient && (typeof httpClient.post === 'function' && typeof httpClient.get === 'function')) {
        return httpClient;
      }
      return axios;
    };
  
    const client = normalizeHttpClient();
  
    const prepareHeaders = (): Record<string, string> => ({
      'Content-Type': 'application/json',
      ...(typeof getRagAuthHeaders === 'function' ? getRagAuthHeaders() : {}),
    });
  
    const postWithRetry = async (endpoint: string, payload: unknown): Promise<AxiosResponse> => {
      let lastError: Error | undefined;
      for (let attempt = 1; attempt <= Math.max(1, maxRetries as number); attempt += 1) {
        try {
          const response = await client.post(endpoint, payload, {
            timeout: (ingestTimeout as number) || DEFAULT_INGEST_TIMEOUT_MS,
            headers: prepareHeaders(),
          });
          return response;
        } catch (error: unknown) {        const err = error as Error & { response?: { status: number } };
        lastError = err;
        const status = err?.response?.status;
        emit('warn', 'rag_service.postFailed', {
          endpoint,
          attempt,
          status,
          error: err?.message,
        });
        if (status && status < 500) {
          break;
        }
        if (attempt < Math.max(1, maxRetries as number)) {
          await wait((retryDelayMs as number) * attempt);
        }
      }
    }
    throw lastError;
  };

  const ingest = async ({ url, content, type, metadata, forceRefresh }: IngestOptions) => {
    return postWithRetry(`${ragServiceUrl}/api/v1/ingest`, {
      url,
      content,
      type,
      metadata: metadata || {},
      force_refresh: Boolean(forceRefresh),
    });
  };

  const ingestCanadaCa = async () => {
    return postWithRetry(`${ragServiceUrl}/api/v1/ingest/canada-ca`, {});
  };

  const getProgressStream = async (url: string) => {
    return client.get(`${ragServiceUrl}/api/v1/ingest/progress`, {
      params: { url },
      responseType: 'stream',
      headers: {
        Accept: 'text/event-stream',
        'Cache-Control': 'no-cache',
        ...prepareHeaders(),
      },
      timeout: (ingestTimeout as number) || DEFAULT_INGEST_TIMEOUT_MS,
    });
  };

  return {
    ingest,
    ingestCanadaCa,
    getProgressStream,
  };
};
