import { cn } from '@/lib/utils';

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'shimmer' | 'pulse';
}

function Skeleton({ className, variant = 'shimmer', ...props }: SkeletonProps) {
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-md bg-muted',
        variant === 'pulse' && 'animate-pulse',
        variant === 'shimmer' && 'skeleton-shimmer',
        className,
      )}
      {...props}
    />
  );
}

// Specialized skeleton components for common UI patterns
export function SkeletonText({
  lines = 1,
  className = '',
}: {
  lines?: number;
  className?: string;
}) {
  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn('h-4', i === lines - 1 && lines > 1 ? 'w-4/5' : 'w-full')}
        />
      ))}
    </div>
  );
}

export function SkeletonButton({
  size = 'default',
  className = '',
}: {
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}) {
  const sizeClasses = {
    sm: 'h-8 w-20',
    default: 'h-10 w-24',
    lg: 'h-12 w-32',
  };

  return <Skeleton className={cn(sizeClasses[size], 'rounded-md', className)} />;
}

export function SkeletonAvatar({
  size = 'default',
  className = '',
}: {
  size?: 'sm' | 'default' | 'lg';
  className?: string;
}) {
  const sizeClasses = {
    sm: 'h-8 w-8',
    default: 'h-10 w-10',
    lg: 'h-16 w-16',
  };

  return <Skeleton className={cn(sizeClasses[size], 'rounded-full', className)} />;
}

export function SkeletonCard({
  className = '',
  showImage = true,
}: {
  className?: string;
  showImage?: boolean;
}) {
  return (
    <div className={cn('space-y-3', className)}>
      {showImage && <Skeleton className="h-48 w-full rounded-lg" />}
      <div className="space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    </div>
  );
}

export function SkeletonChatMessage({
  variant = 'received',
  className = '',
}: {
  variant?: 'sent' | 'received';
  className?: string;
}) {
  return (
    <div
      className={cn('flex items-start gap-3', variant === 'sent' && 'flex-row-reverse', className)}
    >
      <SkeletonAvatar size="sm" />
      <div className={cn('space-y-2', variant === 'sent' ? 'items-end' : 'items-start')}>
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-4 w-36" />
        <Skeleton className="h-3 w-16 opacity-50" />
      </div>
    </div>
  );
}

export function SkeletonSource({ className = '' }: { className?: string }) {
  return (
    <div className={cn('border rounded-lg p-4 space-y-3', className)}>
      <div className="flex items-center gap-2">
        <Skeleton className="h-4 w-4 rounded" />
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-4 w-16 rounded-full" />
      </div>
      <SkeletonText lines={2} />
      <div className="flex items-center gap-4">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-24" />
      </div>
    </div>
  );
}

export function SkeletonTable({
  rows = 3,
  cols = 4,
  className = '',
}: {
  rows?: number;
  cols?: number;
  className?: string;
}) {
  return (
    <div className={cn('space-y-3', className)}>
      {/* Header */}
      <div className="flex gap-3">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} className="h-8 flex-1" />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="flex gap-3">
          {Array.from({ length: cols }).map((_, colIndex) => (
            <Skeleton key={colIndex} className="h-6 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

export { Skeleton };
