import TrendSparkline from './TrendSparkline';
import type { MetricStats } from '@/types/performance';

interface MetricCardProps {
  title: string;
  metric: MetricStats;
  unit?: 'ms' | 'ratio' | 'count';
  description?: string;
  target?: number;
}

const formatLatency = (value: number): string => {
  if (!Number.isFinite(value)) {
    return '–';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`;
  }
  return `${Math.round(value)} ms`;
};

const formatRatio = (value: number): string => {
  if (!Number.isFinite(value)) {
    return '–';
  }
  return `${(value * 100).toFixed(1)}%`;
};

const formatCount = (value: number): string => {
  if (!Number.isFinite(value)) {
    return '–';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }
  return `${Math.round(value)}`;
};

const formatValue = (value: number, unit: MetricCardProps['unit']): string => {
  switch (unit) {
    case 'ratio':
      return formatRatio(value);
    case 'count':
      return formatCount(value);
    case 'ms':
    default:
      return formatLatency(value);
  }
};

const MetricCard = ({ title, metric, unit = 'ms', description, target }: MetricCardProps) => {
  const primary = unit === 'ratio' ? metric.p75 : metric.p95 || metric.mean;
  const formattedPrimary = formatValue(primary, unit);
  const formattedP50 = formatValue(metric.p50, unit);
  const formattedP95 = formatValue(metric.p95, unit);

  const targetLabel = Number.isFinite(target)
    ? unit === 'ratio'
      ? formatRatio(target as number)
      : formatValue(target as number, unit)
    : null;

  return (
    <div className="rounded-xl border border-border/60 bg-background/60 backdrop-blur p-4 flex flex-col gap-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          <p className="text-2xl font-semibold text-foreground mt-1">{formattedPrimary}</p>
          {description && (
            <p className="text-xs text-muted-foreground mt-1 max-w-xs leading-relaxed">
              {description}
            </p>
          )}
        </div>
        <div className="text-right text-xs text-muted-foreground space-y-1">
          <div>
            <span className="uppercase tracking-wide mr-1">P50</span>
            <span>{formattedP50}</span>
          </div>
          <div>
            <span className="uppercase tracking-wide mr-1">P95</span>
            <span>{formattedP95}</span>
          </div>
          {targetLabel && (
            <div>
              <span className="uppercase tracking-wide mr-1">Target</span>
              <span>{targetLabel}</span>
            </div>
          )}
        </div>
      </div>
      <TrendSparkline data={metric.recent} />
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Samples: {metric.count}</span>
        <span>Window: {metric.windowSize}</span>
      </div>
    </div>
  );
};

export default MetricCard;
