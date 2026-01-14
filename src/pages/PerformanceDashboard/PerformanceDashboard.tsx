import { useMemo } from 'react';
import { AlertTriangle, RefreshCcw } from 'lucide-react';
import MetricCard from '@/components/performance/MetricCard';
import TrendSparkline from '@/components/performance/TrendSparkline';
import usePerformanceMetrics from '@/hooks/usePerformanceMetrics';
import { Button } from '@/components/ui/button';
import type { PerformanceMetrics } from '@/types/performance';

const formatPercent = (value?: number) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  return `${(value * 100).toFixed(1)}%`;
};

const formatNumber = (value?: number) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }
  return `${Math.round(value)}`;
};

const PerformanceDashboard = () => {
  const { status, data, error, refresh, isLoading, isError, lastUpdated } = usePerformanceMetrics({
    refreshInterval: 45000,
  });

  const qualitySummary = useMemo(() => {
    if (!data) {
      return null;
    }
    const errorRate = data.quality.errorRate.errorRate;
    const coverage = data.quality.contextCoverage.mean;
    const hallucinations = data.quality.hallucinationRate.mean;
    return {
      errorRate,
      coverage,
      hallucinations,
      failed: data.quality.errorRate.failedRequests,
      total: data.quality.errorRate.totalRequests,
    };
  }, [data]);

  const handleRefresh = () => {
    refresh({ force: true });
  };

  const lastUpdatedLabel = lastUpdated ? new Date(lastUpdated).toLocaleString() : '—';

  const showSkeleton = isLoading && !data;

  return (
    <div className="min-h-screen bg-muted/10 dark:bg-background py-10">
      <div className="max-w-6xl mx-auto px-4 lg:px-0 space-y-10">
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <h1 className="text-3xl font-semibold text-foreground">Performance Dashboard</h1>
            <p className="text-muted-foreground mt-2 max-w-2xl">
              Observe RAG latency, quality, and throughput metrics in real time. Values update
              automatically and can be refreshed on demand.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-xs text-muted-foreground text-right">
              <p className="uppercase tracking-wide">Last updated</p>
              <p className="font-medium">{lastUpdatedLabel}</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isLoading}
              className="gap-2"
            >
              <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </header>

        {isError && (
          <div className="flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-destructive">
            <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium">Failed to fetch metrics</p>
              <p className="text-sm opacity-80">{error}</p>
            </div>
          </div>
        )}

        {showSkeleton && (
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className="h-40 rounded-xl border border-border/60 bg-muted animate-pulse"
              />
            ))}
          </div>
        )}

        {data && (
          <div className="space-y-8">
            <section className="space-y-3">
              <h2 className="text-lg font-semibold text-foreground">Latency</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                <MetricCard
                  title="Answer Time"
                  metric={data.latency.answerTime}
                  description="End-to-end response time"
                  target={2_500}
                />
                <MetricCard
                  title="Search Time"
                  metric={data.latency.searchTime}
                  description="Retrieval pipeline execution"
                  target={500}
                />
                <MetricCard
                  title="Retrieval Assembly"
                  metric={data.latency.retrievalTime}
                  description="Context assembly latency"
                  target={300}
                />
                <MetricCard
                  title="First Token"
                  metric={data.latency.firstToken}
                  description="Streaming initial token latency"
                  target={1_000}
                />
              </div>
            </section>

            <section className="space-y-3">
              <h2 className="text-lg font-semibold text-foreground">Quality</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <MetricCard
                  title="Context Coverage"
                  metric={data.quality.contextCoverage}
                  unit="ratio"
                  description="Share of answers backed by retrieved context"
                  target={0.9}
                />
                <MetricCard
                  title="Support Ratio"
                  metric={data.quality.contextSupport}
                  unit="ratio"
                  description="Context tokens relative to answer tokens"
                  target={0.9}
                />
                <MetricCard
                  title="Answer vs Context"
                  metric={data.quality.answerToContext}
                  unit="ratio"
                  description="Answer size relative to supporting context"
                  target={1.2}
                />
                <MetricCard
                  title="Hallucination Rate"
                  metric={data.quality.hallucinationRate}
                  unit="ratio"
                  description="Responses missing supporting evidence"
                  target={0.05}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <MetricCard
                  title="Answer Tokens"
                  metric={data.quality.answerTokens}
                  unit="count"
                  description="Tokens generated per answer"
                />
                <MetricCard
                  title="Source Tokens"
                  metric={data.quality.sourceTokens}
                  unit="count"
                  description="Tokens included from supporting sources"
                />
                <MetricCard
                  title="Retrieval Score Avg"
                  metric={data.quality.retrievalScores.avg}
                  unit="ratio"
                  description="Mean retriever confidence"
                />
                <MetricCard
                  title="Retrieval Score Gap"
                  metric={data.quality.retrievalScores.gap}
                  unit="ratio"
                  description="Separation between top documents"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-xl border border-border/60 bg-background p-4 flex flex-col gap-4 shadow-sm">
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground">Error Rate</h3>
                    <p className="text-2xl font-semibold text-foreground mt-1">
                      {formatPercent(qualitySummary?.errorRate)}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {`Failures: ${formatNumber(qualitySummary?.failed)} of ${formatNumber(qualitySummary?.total)}`}
                    </p>
                  </div>
                  <TrendSparkline
                    data={data.quality.hallucinationRate.recent}
                    className="text-red-500"
                  />
                </div>
                <div className="rounded-xl border border-border/60 bg-background p-4 flex flex-col gap-4 shadow-sm">
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground">Source Count</h3>
                    <p className="text-2xl font-semibold text-foreground mt-1">
                      {formatNumber(data.quality.sourceCount.p75)}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Median supporting documents referenced per answer
                    </p>
                  </div>
                  <TrendSparkline data={data.quality.sourceCount.recent} className="text-primary" />
                </div>
              </div>
            </section>

            <section className="space-y-3">
              <h2 className="text-lg font-semibold text-foreground">Throughput</h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                <div className="rounded-lg border border-border/50 bg-card p-4">
                  <p className="text-muted-foreground uppercase text-xs">Requests / min</p>
                  <p className="text-xl font-semibold">
                    {formatNumber(data.throughput.requestsPerMinute)}
                  </p>
                </div>
                <div className="rounded-lg border border-border/50 bg-card p-4">
                  <p className="text-muted-foreground uppercase text-xs">Total Requests</p>
                  <p className="text-xl font-semibold">
                    {formatNumber(data.throughput.totalRequests)}
                  </p>
                </div>
                <div className="rounded-lg border border-border/50 bg-card p-4">
                  <p className="text-muted-foreground uppercase text-xs">Successful</p>
                  <p className="text-xl font-semibold">
                    {formatNumber(data.throughput.successfulRequests)}
                  </p>
                </div>
                <div className="rounded-lg border border-border/50 bg-card p-4">
                  <p className="text-muted-foreground uppercase text-xs">Failed</p>
                  <p className="text-xl font-semibold">
                    {formatNumber(data.throughput.failedRequests)}
                  </p>
                </div>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

export default PerformanceDashboard;
