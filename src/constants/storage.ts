export const StorageKeys = {
  theme: 'cf-travel-bot-theme',
  fontSize: 'chatFontSize',
  selectedModel: 'selectedLLMModel',
  selectedProvider: 'selectedLLMProvider',
  whatsNewLastSeen: 'whatsNewLastSeen',
  hybridSearch: 'useHybridSearch',
  visitCount: 'cf-travel-bot-visit-count',
  tabsVariant: 'tabs_ab_variant',
  ingestionHistory: 'ingestionHistory',
  activityLog: 'databaseActivityLog',
  analyticsSessionId: 'cbthis.analytics.sessionId',
  shortAnswerMode: 'shortAnswerMode',
} as const;

export type StorageKey = (typeof StorageKeys)[keyof typeof StorageKeys];
