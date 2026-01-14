import { useCallback, useState } from 'react';
import { toast } from 'sonner';

import { apiClient, ApiError } from '@/api/client';
import { useIngestionHistory } from './useIngestionHistory';
import type { IngestionHistoryEntry } from '../types';

interface UseIngestionControllerOptions {
  activeTab: string;
  onActivityLog: (action: string, details: string) => void;
  refreshDatabaseMetrics: () => Promise<void> | void;
}

const DEFAULT_PROGRESS_ENDPOINT = '/api/rag/ingest/progress';

export const useIngestionController = ({
  activeTab,
  onActivityLog,
  refreshDatabaseMetrics,
}: UseIngestionControllerOptions) => {
  const [urlInput, setUrlInput] = useState('');
  const [isIngesting, setIsIngesting] = useState(false);
  const [forceRefresh, setForceRefresh] = useState(false);
  const [showIngestionProgress, setShowIngestionProgress] = useState(false);
  const [currentIngestionUrl, setCurrentIngestionUrl] = useState('');
  const [ingestionProgressEndpoint, setIngestionProgressEndpoint] = useState<string | null>(
    DEFAULT_PROGRESS_ENDPOINT,
  );
  const { ingestionHistory, recordHistoryEntry, clearIngestionHistory } = useIngestionHistory();

  const resetIngestionProgress = useCallback(() => {
    setShowIngestionProgress(false);
    setCurrentIngestionUrl('');
    setIngestionProgressEndpoint(DEFAULT_PROGRESS_ENDPOINT);
  }, []);

  const handleIngestionProgressComplete = useCallback(() => {
    resetIngestionProgress();
    if (activeTab === 'database') {
      void refreshDatabaseMetrics();
    }
  }, [activeTab, refreshDatabaseMetrics, resetIngestionProgress]);

  const handleIngestURL = useCallback(async () => {
    if (!urlInput.trim()) {
      toast.error('Please enter a URL');
      return;
    }

    try {
      new URL(urlInput.trim());
    } catch {
      toast.error('Please enter a valid URL');
      return;
    }

    const normalizedUrl = urlInput.trim();

    setIsIngesting(true);

    // Show progress UI immediately before starting ingestion
    setCurrentIngestionUrl(normalizedUrl);
    setIngestionProgressEndpoint('/api/v2/ingest/progress');
    setShowIngestionProgress(true);

    try {
      const ingestionTargets = [
        { submit: '/api/v2/ingest', progress: '/api/v2/ingest/progress' },
        { submit: '/api/rag/ingest', progress: '/api/rag/ingest/progress' },
      ] as const;

      let targetUsed: (typeof ingestionTargets)[number] | null = null;
      let lastError: string | null = null;
      let responseData: unknown = null;
      let responseOk = false;

      for (const target of ingestionTargets) {
        try {
          responseData = await apiClient.postJson(
            target.submit,
            {
              url: normalizedUrl,
              type: 'web',
              forceRefresh,
              metadata: {
                source: 'manual_ingestion',
                ingested_from: 'config_page',
              },
            },
            {
              headers: {
                'Content-Type': 'application/json',
              },
            },
          );
          responseOk = true;
          targetUsed = target;
          break;
        } catch (error) {
          if (error instanceof ApiError) {
            const data = error.data as Record<string, unknown> | undefined;
            const message =
              typeof data?.message === 'string' ? data.message : error.statusText || error.message;
            lastError = message;
            responseData = error.data;
            if (error.status === 404) {
              continue;
            }
          } else {
            console.error('Ingestion request error:', error);
            lastError = 'Network error during ingestion';
            responseData = null;
          }
        }
      }

      if (!targetUsed) {
        toast.error(lastError || 'Unable to reach ingestion service');
        resetIngestionProgress();
        return;
      }

      const data = (responseData as Record<string, unknown>) || {};
      // Update progress endpoint if different target was used
      if (targetUsed.progress && targetUsed.progress !== ingestionProgressEndpoint) {
        setIngestionProgressEndpoint(targetUsed.progress);
      }

      if (responseOk) {
        if (data.status === 'success') {
          toast.success(`Successfully ingested ${data.chunks_created as number} chunks from URL`);
        } else if (data.status === 'exists') {
          toast.info('Document already exists in the database. Use force refresh to re-ingest.');
          resetIngestionProgress();
        } else {
          toast.info(
            (data.message as string) || 'Ingestion request received. Monitoring progress...',
          );
        }

        const historyEntry: IngestionHistoryEntry = {
          url: normalizedUrl,
          status: (data.status === 'exists' ? 'exists' : (data.status as string)) || 'pending',
          timestamp: new Date().toISOString(),
        };
        recordHistoryEntry(historyEntry);

        onActivityLog(
          'Document Ingested',
          `${normalizedUrl} - ${(data.chunks_created as number) ?? 0} chunks`,
        );

        setUrlInput('');
        setForceRefresh(false);

        if (activeTab === 'database') {
          void refreshDatabaseMetrics();
        }
      } else {
        const errorMessage = (data?.message as string) || lastError || 'Failed to ingest URL';
        toast.error(errorMessage);
        resetIngestionProgress();

        recordHistoryEntry({
          url: normalizedUrl,
          status: 'failed',
          timestamp: new Date().toISOString(),
        });
      }
    } catch (error) {
      console.error('Ingestion error:', error);
      toast.error('Network error during ingestion');
      resetIngestionProgress();
    } finally {
      setIsIngesting(false);
    }
  }, [
    activeTab,
    forceRefresh,
    ingestionProgressEndpoint,
    onActivityLog,
    recordHistoryEntry,
    refreshDatabaseMetrics,
    resetIngestionProgress,
    urlInput,
  ]);

  return {
    urlInput,
    setUrlInput,
    isIngesting,
    forceRefresh,
    setForceRefresh,
    ingestionHistory,
    showIngestionProgress,
    currentIngestionUrl,
    ingestionProgressEndpoint,
    handleIngestURL,
    handleIngestionProgressComplete,
    clearIngestionHistory,
  };
};
