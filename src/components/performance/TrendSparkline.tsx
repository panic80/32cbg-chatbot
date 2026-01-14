import type { MetricSample } from '@/types/performance';

interface TrendSparklineProps {
  data: MetricSample[];
  height?: number;
  className?: string;
}

const TrendSparkline = ({
  data,
  height = 48,
  className = 'text-blue-500 dark:text-blue-300',
}: TrendSparklineProps) => {
  if (!data || data.length === 0) {
    return (
      <div
        className="h-12 w-full flex items-center justify-center text-xs text-muted-foreground"
        aria-label="No trend data"
      >
        No data
      </div>
    );
  }

  const values = data.map((sample) => sample.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const width = Math.max(data.length - 1, 1);

  const points = data
    .map((sample, index) => {
      const x = (index / width) * 100;
      const y = ((max - sample.value) / span) * 100;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className={`w-full ${className}`}
      style={{ height }}
      role="img"
      aria-label="Trend sparkline"
    >
      <polyline fill="none" strokeWidth={2} stroke="currentColor" points={points} />
    </svg>
  );
};

export default TrendSparkline;
