import { useState, useCallback, useEffect } from 'react';
import { apiClient } from '@/api/client';
import { toast } from 'sonner';

export interface RAGConfig {
  enable_hyde: boolean;
  hyde_model: string;
  hyde_timeout: number;
  enable_query_logging: boolean;
}

interface RAGConfigResponse {
  status: string;
  config: RAGConfig;
}

interface RAGConfigUpdateResponse {
  status: string;
  updated: string[];
  restart_required: boolean;
  restart_required_for: string[];
}

const DEFAULT_CONFIG: RAGConfig = {
  enable_hyde: false,
  hyde_model: 'gpt-4.1-mini',
  hyde_timeout: 2.0,
  enable_query_logging: true,
};

/**
 * Hook for managing RAG service runtime configuration (hot-toggleable settings)
 */
export const useRAGConfig = () => {
  const [config, setConfig] = useState<RAGConfig>(DEFAULT_CONFIG);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadConfig = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.getJson<RAGConfigResponse>('/api/admin/rag/config');
      if (response.status === 'success' && response.config) {
        setConfig(response.config);
      }
    } catch (err) {
      console.error('Failed to load RAG config:', err);
      setError('Failed to load RAG configuration');
      // Use defaults on error
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load config from RAG service on mount
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const updateConfig = useCallback(async (updates: Partial<RAGConfig>) => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await apiClient.postJson<RAGConfigUpdateResponse>('/api/admin/rag/config', {
        config_updates: updates,
      });

      if (response.status === 'success') {
        // Update local state
        setConfig((prev) => ({ ...prev, ...updates }));
        toast.success('RAG configuration updated');

        if (response.restart_required) {
          toast.warning(`Restart required for: ${response.restart_required_for.join(', ')}`);
        }

        return true;
      }

      throw new Error('Update failed');
    } catch (err) {
      console.error('Failed to update RAG config:', err);
      setError('Failed to update RAG configuration');
      toast.error('Failed to update RAG configuration');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, []);

  const toggleHyDE = useCallback(
    async (enabled: boolean) => {
      return updateConfig({ enable_hyde: enabled });
    },
    [updateConfig],
  );

  return {
    config,
    isLoading,
    isSaving,
    error,

    // Actions
    loadConfig,
    updateConfig,
    toggleHyDE,

    // Convenience accessors
    hydeEnabled: config.enable_hyde,
    hydeModel: config.hyde_model,
  };
};

export default useRAGConfig;
