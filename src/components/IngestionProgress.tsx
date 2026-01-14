import React, { useEffect, useState, useCallback } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Progress } from './ui/progress';
import { Card } from './ui/card';
import {
  CheckCircle2,
  Circle,
  Loader2,
  AlertCircle,
  FileText,
  Split,
  Brain,
  Database,
  Filter,
} from 'lucide-react';
import { cn } from '../lib/utils';

interface IngestionStep {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  message?: string;
  progress?: number;
  startTime?: number;
  endTime?: number;
  details?: {
    current?: number;
    total?: number;
    rate?: number;
  };
}

interface IngestionProgressProps {
  isOpen: boolean;
  onClose: () => void;
  url: string;
  operationId?: string;
}

interface InlineIngestionProgressProps {
  url: string;
  progressEndpoint?: string;
  onComplete?: (success: boolean) => void;
  className?: string;
}

const stepIcons: Record<string, React.ReactNode> = {
  loading: <FileText className="h-4 w-4" />,
  splitting: <Split className="h-4 w-4" />,
  deduplicating: <Filter className="h-4 w-4" />,
  embedding: <Brain className="h-4 w-4" />,
  storing: <Database className="h-4 w-4" />,
};

const initialSteps: IngestionStep[] = [
  { id: 'loading', name: 'Loading document', status: 'pending' },
  { id: 'splitting', name: 'Splitting into chunks', status: 'pending' },
  { id: 'deduplicating', name: 'Deduplicating chunks', status: 'pending' },
  { id: 'embedding', name: 'Generating embeddings', status: 'pending' },
  { id: 'storing', name: 'Storing in vector database', status: 'pending' },
];

// Hook to manage progress state and SSE connection
function useIngestionProgress(url: string, progressEndpoint?: string, enabled: boolean = true) {
  const [steps, setSteps] = useState<IngestionStep[]>(initialSteps.map((s) => ({ ...s })));
  const [overallProgress, setOverallProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const handleProgressUpdate = useCallback(
    (
      data: {
        type: string;
        stepId?: string;
        message?: string;
        progress?: number;
        details?: Record<string, unknown>;
      },
      eventSourceRef: { current: EventSource | null },
    ) => {
      switch (data.type) {
        case 'connected':
          setIsConnected(true);
          break;

        case 'step_start':
          setSteps((prev) =>
            prev.map((step) =>
              step.id === data.stepId
                ? {
                    ...step,
                    status: 'in_progress',
                    startTime: Date.now(),
                    message: data.message,
                    progress: 0,
                  }
                : step,
            ),
          );
          break;

        case 'step_progress':
          setSteps((prev) =>
            prev.map((step) =>
              step.id === data.stepId
                ? {
                    ...step,
                    status: step.status === 'pending' ? 'in_progress' : step.status,
                    startTime: step.status === 'pending' ? Date.now() : step.startTime,
                    progress: data.progress,
                    message: data.message,
                    details: data.details,
                  }
                : step,
            ),
          );
          break;

        case 'step_complete':
          setSteps((prev) =>
            prev.map((step) =>
              step.id === data.stepId
                ? {
                    ...step,
                    status: 'completed',
                    endTime: Date.now(),
                    message: data.message,
                    progress: 100,
                  }
                : step,
            ),
          );
          break;

        case 'step_error':
          setSteps((prev) =>
            prev.map((step) =>
              step.id === data.stepId ? { ...step, status: 'error', message: data.message } : step,
            ),
          );
          setError(data.message);
          break;

        case 'overall_progress':
          setOverallProgress(Math.round(data.progress));
          break;

        case 'complete':
          setIsComplete(true);
          setOverallProgress(100);
          eventSourceRef.current?.close();
          break;

        case 'error':
          setError(data.message);
          eventSourceRef.current?.close();
          break;
      }
    },
    [],
  );

  useEffect(() => {
    if (!enabled || !url) return;

    // Reset state when starting new ingestion
    setSteps(initialSteps.map((s) => ({ ...s })));
    setOverallProgress(0);
    setIsComplete(false);
    setError(null);
    setIsConnected(false);

    const baseEndpoint = progressEndpoint ?? '/api/v2/ingest/progress';
    const separator = baseEndpoint.includes('?') ? '&' : '?';
    const eventSourceUrl = `${baseEndpoint}${separator}url=${encodeURIComponent(url)}`;

    const eventSourceRef = { current: null as EventSource | null };
    const es = new EventSource(eventSourceUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      setIsConnected(true);
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleProgressUpdate(data, eventSourceRef);
      } catch (err) {
        console.error('Failed to parse progress update:', err);
      }
    };

    es.onerror = () => {
      setIsConnected(false);
      es.close();
    };

    return () => {
      es.close();
    };
  }, [enabled, url, progressEndpoint, handleProgressUpdate]);

  return { steps, overallProgress, isComplete, error, isConnected };
}

// Shared helper functions
const getStepIcon = (step: IngestionStep) => {
  if (step.status === 'completed') {
    return <CheckCircle2 className="h-4 w-4 text-green-600" />;
  } else if (step.status === 'in_progress') {
    return <Loader2 className="h-4 w-4 animate-spin text-blue-600" />;
  } else if (step.status === 'error') {
    return <AlertCircle className="h-4 w-4 text-red-600" />;
  } else {
    return <Circle className="h-4 w-4 text-gray-400" />;
  }
};

const formatDuration = (startTime?: number, endTime?: number) => {
  if (!startTime) return '';
  const end = endTime || Date.now();
  const duration = (end - startTime) / 1000;
  return `${duration.toFixed(1)}s`;
};

const formatRate = (rate?: number, unit: string = 'items/s') => {
  if (!rate) return '';
  return `${rate.toFixed(1)} ${unit}`;
};

// Shared progress content component
function ProgressContent({
  steps,
  overallProgress,
  isComplete,
  error,
  url,
  showUrl = true,
  compact = false,
}: {
  steps: IngestionStep[];
  overallProgress: number;
  isComplete: boolean;
  error: string | null;
  url?: string;
  showUrl?: boolean;
  compact?: boolean;
}) {
  return (
    <div className={cn('space-y-4', compact && 'space-y-3')}>
      {/* URL being processed */}
      {showUrl && url && (
        <div className="p-3 bg-muted rounded-lg">
          <p className="text-sm font-medium">Processing URL:</p>
          <p className="text-xs text-muted-foreground truncate">{url}</p>
        </div>
      )}

      {/* Overall progress */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="font-medium">Overall Progress</span>
          <span className="tabular-nums">{overallProgress}%</span>
        </div>
        <Progress value={overallProgress} className="h-2" />
      </div>

      {/* Individual steps */}
      <div className={cn('space-y-3', compact && 'space-y-2')}>
        {steps.map((step) => (
          <div key={step.id} className="space-y-1">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex-shrink-0">{getStepIcon(step)}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <p
                    className={cn(
                      'text-sm font-medium transition-colors duration-200',
                      step.status === 'completed' && 'text-green-600',
                      step.status === 'error' && 'text-red-600',
                      step.status === 'in_progress' && 'text-blue-600',
                    )}
                  >
                    {step.name}
                  </p>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {step.details &&
                      step.details.current !== undefined &&
                      step.details.total !== undefined && (
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {step.details.current}/{step.details.total}
                        </span>
                      )}
                    {step.details?.rate !== undefined && (
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {formatRate(
                          step.details.rate,
                          step.id === 'embedding'
                            ? 'emb/s'
                            : step.id === 'splitting'
                              ? 'ch/s'
                              : step.id === 'deduplicating'
                                ? 'ck/s'
                                : '/s',
                        )}
                      </span>
                    )}
                    {step.startTime && (
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {formatDuration(step.startTime, step.endTime)}
                      </span>
                    )}
                  </div>
                </div>

                {step.status === 'in_progress' && step.progress !== undefined && (
                  <Progress value={step.progress} className="h-1 mt-1.5" />
                )}
              </div>
              <div className="text-muted-foreground flex-shrink-0">{stepIcons[step.id]}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Error message */}
      {error && (
        <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Success message */}
      {isComplete && !error && (
        <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <p className="text-sm text-green-600 dark:text-green-400">
            Document successfully ingested!
          </p>
        </div>
      )}
    </div>
  );
}

// Dialog version (original)
export default function IngestionProgress({ isOpen, onClose, url }: IngestionProgressProps) {
  const { steps, overallProgress, isComplete, error } = useIngestionProgress(
    url,
    undefined,
    isOpen,
  );

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Document Ingestion Progress</DialogTitle>
        </DialogHeader>
        <ProgressContent
          steps={steps}
          overallProgress={overallProgress}
          isComplete={isComplete}
          error={error}
          url={url}
        />
      </DialogContent>
    </Dialog>
  );
}

// Inline version (new)
export function InlineIngestionProgress({
  url,
  progressEndpoint,
  onComplete,
  className,
}: InlineIngestionProgressProps) {
  const { steps, overallProgress, isComplete, error, isConnected } = useIngestionProgress(
    url,
    progressEndpoint,
    true,
  );

  // Call onComplete when ingestion finishes
  useEffect(() => {
    if (isComplete && onComplete) {
      onComplete(!error);
    }
  }, [isComplete, error, onComplete]);

  return (
    <Card className={cn('p-4', className)}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold">Ingestion Progress</h3>
        <div className="flex items-center gap-2 text-xs">
          <div
            className={cn(
              'flex items-center gap-1',
              isConnected ? 'text-green-600' : 'text-muted-foreground',
            )}
          >
            <div
              className={cn(
                'h-2 w-2 rounded-full transition-colors duration-300',
                isConnected ? 'bg-green-600' : 'bg-muted-foreground',
              )}
            />
            {isConnected ? 'Live' : 'Connecting...'}
          </div>
        </div>
      </div>
      <ProgressContent
        steps={steps}
        overallProgress={overallProgress}
        isComplete={isComplete}
        error={error}
        showUrl={false}
        compact
      />
    </Card>
  );
}
