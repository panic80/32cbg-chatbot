import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  DEFAULT_INGEST_TIMEOUT_MS,
  DEFAULT_MAX_RETRIES,
  DEFAULT_RETRY_DELAY_MS,
  RAG_SERVICE_URL,
} from '../config/constants.js';
import { createScopedLogger } from '../utils/loggerHelpers.js';
import { withRetry } from './retryService.js';

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

  const { emit } = createScopedLogger(logger, 'service:rag');

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
    return withRetry(
      async () => {
        return client.post(endpoint, payload, {
          timeout: (ingestTimeout as number) || DEFAULT_INGEST_TIMEOUT_MS,
          headers: prepareHeaders(),
        });
      },
      {
        maxRetries: maxRetries as number,
        initialDelayMs: retryDelayMs as number,
        emit,
        logScope: 'rag_service',
      },
    );
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
