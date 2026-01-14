export interface MetricSample {
  value: number;
  timestamp: string;
}

export interface MetricStats {
  count: number;
  mean: number;
  min: number;
  max: number;
  p50: number;
  p75: number;
  p95: number;
  p99: number;
  ratePerMinute: number;
  windowSize: number;
  recent: MetricSample[];
}

export interface ErrorRateSummary {
  totalRequests: number;
  failedRequests: number;
  errorRate: number;
  errorsByType: Record<string, number>;
}

export interface LatencyMetrics {
  answerTime: MetricStats;
  searchTime: MetricStats;
  retrievalTime: MetricStats;
  answerGeneration: MetricStats;
  firstToken: MetricStats;
}

export interface RetrievalScoreMetrics {
  avg: MetricStats;
  max: MetricStats;
  min: MetricStats;
  std: MetricStats;
  gap: MetricStats;
}

export interface QualityMetrics {
  contextCoverage: MetricStats;
  contextSupport: MetricStats;
  answerToContext: MetricStats;
  hallucinationRate: MetricStats;
  answerTokens: MetricStats;
  sourceTokens: MetricStats;
  sourceCount: MetricStats;
  retrievalScores: RetrievalScoreMetrics;
  errorRate: ErrorRateSummary;
}

export interface ThroughputMetrics {
  requestsPerMinute?: number;
  totalRequests?: number;
  successfulRequests?: number;
  failedRequests?: number;
}

export interface GatewayMeta {
  cached?: boolean;
  fetchedAt?: string;
  ragEndpoint?: string;
}

export interface PerformanceMeta {
  windowSize?: number;
  updatedAt?: string;
  uptimeSeconds?: number;
  [key: string]: unknown;
}

export interface PerformanceMetrics {
  latency: LatencyMetrics;
  quality: QualityMetrics;
  throughput: ThroughputMetrics;
  cache: Record<string, unknown>;
  retrievers: Record<string, unknown>;
  tokenUsage: Record<string, unknown>;
  meta: PerformanceMeta;
  gatewayMeta?: GatewayMeta;
}

export interface PerformanceState {
  status: 'idle' | 'loading' | 'success' | 'error';
  data: PerformanceMetrics | null;
  error?: string;
}
