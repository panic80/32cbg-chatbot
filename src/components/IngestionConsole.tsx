import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Card } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { CheckCircle2, Circle, Loader2, AlertCircle, Terminal } from 'lucide-react';
import { cn } from '../lib/utils';

interface LogEntry {
  id: string;
  timestamp: Date;
  type: 'info' | 'success' | 'error' | 'warning' | 'progress';
  message: string;
  details?: unknown;
}

interface ProgressData {
  type: string;
  stepId?: string;
  message?: string;
  progress?: number;
  details?: {
    current?: number;
    total?: number;
    rate?: number;
  };
}

interface IngestionConsoleProps {
  url: string;
  progressEndpoint?: string;
  onComplete?: (success: boolean) => void;
  className?: string;
}

export default function IngestionConsole({
  url,
  progressEndpoint,
  onComplete,
  className,
}: IngestionConsoleProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const addLog = useCallback((type: LogEntry['type'], message: string, details?: unknown) => {
    const entry: LogEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      type,
      message,
      details,
    };
    setLogs((prev) => [...prev, entry]);
  }, []);

  const handleProgressUpdate = useCallback(
    (data: ProgressData) => {
      switch (data.type) {
        case 'connected':
          addLog('success', 'Progress tracking initialized');
          break;

        case 'step_start':
          setCurrentStep(data.stepId || null);
          addLog('info', `[${data.stepId}] ${data.message || 'Started'}`);
          break;

        case 'step_progress':
          if (data.details) {
            const { current, total, rate } = data.details;
            let progressMsg = `[${data.stepId}] Progress: ${data.progress?.toFixed(0)}%`;
            if (current && total) {
              progressMsg += ` (${current}/${total})`;
            }
            if (rate) {
              progressMsg += ` - ${rate.toFixed(1)}/s`;
            }
            addLog('progress', progressMsg);
          } else {
            addLog('progress', `[${data.stepId}] ${data.message}`);
          }
          break;

        case 'step_complete':
          addLog('success', `[${data.stepId}] ${data.message || 'Completed'}`);
          setCurrentStep(null);
          break;

        case 'step_error':
          addLog('error', `[${data.stepId}] Error: ${data.message}`);
          setCurrentStep(null);
          break;

        case 'overall_progress':
          // Don't log overall progress to avoid clutter
          break;

        case 'complete':
          addLog('success', '✓ Ingestion completed successfully!');
          if (onComplete) onComplete(true);
          eventSourceRef.current?.close();
          break;

        case 'error':
          addLog('error', `✗ Ingestion failed: ${data.message}`);
          if (onComplete) onComplete(false);
          eventSourceRef.current?.close();
          break;

        default:
          addLog('info', `[${data.type}] ${JSON.stringify(data)}`);
      }
    },
    [addLog, onComplete],
  );

  useEffect(() => {
    // Initial log
    addLog('info', `Starting ingestion for: ${url}`);

    // Connect to SSE endpoint
    const connectToProgress = () => {
      addLog('info', 'Connecting to progress stream...');

      const baseEndpoint = progressEndpoint ?? '/api/rag/ingest/progress';
      const separator = baseEndpoint.includes('?') ? '&' : '?';
      const eventSourceUrl = `${baseEndpoint}${separator}url=${encodeURIComponent(url)}`;

      const es = new EventSource(eventSourceUrl);
      eventSourceRef.current = es;

      es.onopen = () => {
        setIsConnected(true);
        addLog('success', 'Connected to progress stream');
      };

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleProgressUpdate(data);
        } catch (err) {
          addLog('error', `Failed to parse progress data: ${err}`);
        }
      };

      es.onerror = (error) => {
        setIsConnected(false);
        addLog('error', 'Lost connection to progress stream');
        es.close();

        // Retry connection after 2 seconds
        setTimeout(() => {
          addLog('info', 'Attempting to reconnect...');
          connectToProgress();
        }, 2000);
      };
    };

    connectToProgress();

    // Cleanup
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [progressEndpoint, url, addLog, handleProgressUpdate]);

  const getLogIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="h-3 w-3 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-3 w-3 text-red-600" />;
      case 'warning':
        return <AlertCircle className="h-3 w-3 text-yellow-600" />;
      case 'progress':
        return <Loader2 className="h-3 w-3 animate-spin text-blue-600" />;
      default:
        return <Circle className="h-3 w-3 text-gray-400" />;
    }
  };

  const getLogColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'success':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      case 'warning':
        return 'text-yellow-600';
      case 'progress':
        return 'text-blue-600';
      default:
        return 'text-gray-600';
    }
  };

  const formatTimestamp = (date: Date) => {
    const base = date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    const milliseconds = date.getMilliseconds().toString().padStart(3, '0');
    return `${base}.${milliseconds}`;
  };

  return (
    <Card className={cn('w-full', className)}>
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            <h3 className="font-semibold">Ingestion Console</h3>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div
              className={cn(
                'flex items-center gap-1',
                isConnected ? 'text-green-600' : 'text-red-600',
              )}
            >
              <div
                className={cn('h-2 w-2 rounded-full', isConnected ? 'bg-green-600' : 'bg-red-600')}
              />
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </div>
      </div>

      <ScrollArea className="h-[400px] w-full">
        <div ref={scrollRef} className="p-4 font-mono text-xs space-y-1">
          {logs.map((log) => (
            <div
              key={log.id}
              className="flex items-start gap-2 hover:bg-muted/50 px-2 py-0.5 rounded"
            >
              <span className="text-muted-foreground flex-shrink-0">
                [{formatTimestamp(log.timestamp)}]
              </span>
              <span className="flex-shrink-0">{getLogIcon(log.type)}</span>
              <span className={cn('break-all', getLogColor(log.type))}>{log.message}</span>
            </div>
          ))}

          {currentStep && (
            <div className="flex items-center gap-2 mt-2 px-2">
              <Loader2 className="h-3 w-3 animate-spin text-blue-600" />
              <span className="text-blue-600 text-xs">Processing: {currentStep}...</span>
            </div>
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}
