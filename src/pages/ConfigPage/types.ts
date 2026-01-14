export type ModelProvider = 'openai' | 'google' | 'anthropic' | 'openrouter';

export type ModelDesignation = 'fast' | 'smart';

export type ModelConfig = {
  provider: ModelProvider;
  model: string;
};

export type OperationModelConfig = {
  responseGeneration: ModelDesignation;
  hydeExpansion: ModelDesignation;
  queryRewriting: ModelDesignation;
  followUpGeneration: ModelDesignation;
};

export type FullModelConfig = {
  fastModel: ModelConfig;
  smartModel: ModelConfig;
  operationModels: OperationModelConfig;
  updatedAt?: string;
};

export type OpenRouterModel = {
  id: string;
  name: string;
  description: string;
  contextLength: number;
  isOpenSource: boolean;
  pricing: {
    prompt: string;
    completion: string;
  } | null;
};

export type OpenRouterModelsResponse = {
  models: OpenRouterModel[];
  total: number;
  openSourceCount: number;
  isConfigured: boolean;
};

export type DatabaseStats = {
  totalDocuments: number;
  totalChunks: number;
  totalSources: number;
  lastIngestedAt: string | null;
};

export type DatabaseSource = {
  id: string;
  label: string;
  canonicalUrl: string | null;
  chunkCount: number;
  documentCount: number;
  lastIngestedAt: string | null;
  searchText: string;
};

export type ActivityLogEntry = {
  timestamp: string;
  action: string;
  details: string;
};

export type IngestionHistoryEntry = {
  url: string;
  status: string;
  timestamp: string;
};

export type ChatLogEntry = {
  id: number;
  askedAt: string;
  question: string;
  answer: string | null;
  conversationId: string | null;
  model: string | null;
  provider: string | null;
  ragEnabled: boolean | null;
  shortAnswerMode: boolean | null;
  metadata: unknown;
};

export type LogFilters = {
  search: string;
  startAt: string;
  endAt: string;
  provider: string;
  model: string;
  conversationId: string;
  ragEnabled: 'all' | 'true' | 'false';
  shortAnswerMode: 'all' | 'true' | 'false';
};

export const LOGS_PAGE_SIZE = 20;

export const LOGS_FILTER_DEFAULTS: LogFilters = {
  search: '',
  startAt: '',
  endAt: '',
  provider: 'all',
  model: '',
  conversationId: '',
  ragEnabled: 'all',
  shortAnswerMode: 'all',
};
