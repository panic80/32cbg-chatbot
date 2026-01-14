import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import PerformanceDashboard from '../PerformanceDashboard';
import type { PerformanceMetrics } from '@/types/performance';

defineGlobalMocks();

const metricsSample: PerformanceMetrics = {
  latency: {
    answerTime: {
      count: 5,
      mean: 1500,
      min: 1000,
      max: 2000,
      p50: 1400,
      p75: 1600,
      p95: 1900,
      p99: 2100,
      ratePerMinute: 2,
      windowSize: 5,
      recent: [],
    },
    searchTime: {
      count: 5,
      mean: 400,
      min: 200,
      max: 600,
      p50: 380,
      p75: 420,
      p95: 500,
      p99: 520,
      ratePerMinute: 2,
      windowSize: 5,
      recent: [],
    },
    retrievalTime: {
      count: 5,
      mean: 250,
      min: 120,
      max: 400,
      p50: 220,
      p75: 260,
      p95: 320,
      p99: 350,
      ratePerMinute: 2,
      windowSize: 5,
      recent: [],
    },
    answerGeneration: {
      count: 5,
      mean: 900,
      min: 600,
      max: 1100,
      p50: 850,
      p75: 930,
      p95: 1080,
      p99: 1100,
      ratePerMinute: 2,
      windowSize: 5,
      recent: [],
    },
    firstToken: {
      count: 5,
      mean: 450,
      min: 250,
      max: 600,
      p50: 420,
      p75: 480,
      p95: 560,
      p99: 580,
      ratePerMinute: 2,
      windowSize: 5,
      recent: [],
    },
  },
  quality: {
    contextCoverage: {
      count: 5,
      mean: 0.92,
      min: 0.9,
      max: 0.95,
      p50: 0.92,
      p75: 0.93,
      p95: 0.95,
      p99: 0.95,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    contextSupport: {
      count: 5,
      mean: 0.88,
      min: 0.85,
      max: 0.9,
      p50: 0.87,
      p75: 0.89,
      p95: 0.9,
      p99: 0.9,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    answerToContext: {
      count: 5,
      mean: 1.2,
      min: 1.1,
      max: 1.3,
      p50: 1.2,
      p75: 1.22,
      p95: 1.28,
      p99: 1.3,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    hallucinationRate: {
      count: 5,
      mean: 0.04,
      min: 0.03,
      max: 0.06,
      p50: 0.04,
      p75: 0.05,
      p95: 0.06,
      p99: 0.06,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    answerTokens: {
      count: 5,
      mean: 200,
      min: 160,
      max: 240,
      p50: 190,
      p75: 210,
      p95: 230,
      p99: 240,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    sourceTokens: {
      count: 5,
      mean: 260,
      min: 200,
      max: 320,
      p50: 250,
      p75: 270,
      p95: 310,
      p99: 320,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    sourceCount: {
      count: 5,
      mean: 3,
      min: 2,
      max: 4,
      p50: 3,
      p75: 3.5,
      p95: 4,
      p99: 4,
      ratePerMinute: 0,
      windowSize: 5,
      recent: [],
    },
    retrievalScores: {
      avg: {
        count: 5,
        mean: 0.72,
        min: 0.6,
        max: 0.85,
        p50: 0.7,
        p75: 0.75,
        p95: 0.83,
        p99: 0.85,
        ratePerMinute: 0,
        windowSize: 5,
        recent: [],
      },
      max: {
        count: 5,
        mean: 0.92,
        min: 0.9,
        max: 0.95,
        p50: 0.91,
        p75: 0.93,
        p95: 0.95,
        p99: 0.95,
        ratePerMinute: 0,
        windowSize: 5,
        recent: [],
      },
      min: {
        count: 5,
        mean: 0.42,
        min: 0.35,
        max: 0.5,
        p50: 0.4,
        p75: 0.45,
        p95: 0.5,
        p99: 0.5,
        ratePerMinute: 0,
        windowSize: 5,
        recent: [],
      },
      std: {
        count: 5,
        mean: 0.11,
        min: 0.08,
        max: 0.14,
        p50: 0.1,
        p75: 0.12,
        p95: 0.13,
        p99: 0.14,
        ratePerMinute: 0,
        windowSize: 5,
        recent: [],
      },
      gap: {
        count: 5,
        mean: 0.18,
        min: 0.12,
        max: 0.24,
        p50: 0.17,
        p75: 0.2,
        p95: 0.23,
        p99: 0.24,
        ratePerMinute: 0,
        windowSize: 5,
        recent: [],
      },
    },
    errorRate: {
      totalRequests: 50,
      failedRequests: 2,
      errorRate: 0.04,
      errorsByType: { llm: 1, retrieval: 1 },
    },
  },
  throughput: {
    requestsPerMinute: 3.5,
    totalRequests: 50,
    successfulRequests: 48,
    failedRequests: 2,
  },
  cache: {},
  retrievers: {},
  tokenUsage: {},
  meta: { windowSize: 100, updatedAt: '2024-01-01T00:00:00Z' },
  gatewayMeta: {
    cached: false,
    fetchedAt: '2024-01-01T00:00:10Z',
    ragEndpoint: 'http://rag/api/v1/metrics/summary',
  },
};

vi.mock('@/hooks/usePerformanceMetrics', () => ({
  __esModule: true,
  default: () => ({
    status: 'success',
    data: metricsSample,
    error: undefined,
    isLoading: false,
    isError: false,
    refresh: vi.fn(),
    lastUpdated: metricsSample.gatewayMeta?.fetchedAt,
  }),
}));

function defineGlobalMocks() {
  if (!(global as { ResizeObserver?: unknown }).ResizeObserver) {
    (global as { ResizeObserver?: unknown }).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
}

describe('PerformanceDashboard', () => {
  it('renders metrics summary', () => {
    render(<PerformanceDashboard />);

    expect(screen.getByText('Performance Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Answer Time')).toBeInTheDocument();
    expect(screen.getByText(/Context Coverage/i)).toBeInTheDocument();
    expect(screen.getByText(/Support Ratio/i)).toBeInTheDocument();
    expect(screen.getByText(/Requests \//i)).toBeInTheDocument();
  });
});
