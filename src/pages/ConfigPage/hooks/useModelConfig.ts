import { useState, useCallback, useEffect } from 'react';
import { apiClient } from '@/api/client';
import { toast } from 'sonner';
import type {
  ModelConfig,
  OperationModelConfig,
  FullModelConfig,
  ModelDesignation,
} from '../types';

const DEFAULT_CONFIG: FullModelConfig = {
  fastModel: { provider: 'openai', model: 'gpt-4.1-mini' },
  smartModel: { provider: 'openai', model: 'gpt-5-mini' },
  operationModels: {
    responseGeneration: 'smart',
    hydeExpansion: 'fast',
    queryRewriting: 'fast',
    followUpGeneration: 'fast',
  },
};

const STORAGE_KEY = 'modelConfig';

/**
 * Hook for managing Fast/Smart model configuration
 */
export const useModelConfig = () => {
  const [config, setConfig] = useState<FullModelConfig>(DEFAULT_CONFIG);
  const [savedConfig, setSavedConfig] = useState<FullModelConfig>(DEFAULT_CONFIG);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check if there are unsaved changes
  const hasUnsavedChanges =
    JSON.stringify(config.fastModel) !== JSON.stringify(savedConfig.fastModel) ||
    JSON.stringify(config.smartModel) !== JSON.stringify(savedConfig.smartModel) ||
    JSON.stringify(config.operationModels) !== JSON.stringify(savedConfig.operationModels);

  const loadConfig = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.getJson<FullModelConfig>('/api/admin/model-config');

      const loadedConfig: FullModelConfig = {
        fastModel: response.fastModel || DEFAULT_CONFIG.fastModel,
        smartModel: response.smartModel || DEFAULT_CONFIG.smartModel,
        operationModels: {
          ...DEFAULT_CONFIG.operationModels,
          ...(response.operationModels || {}),
        },
        updatedAt: response.updatedAt,
      };

      setConfig(loadedConfig);
      setSavedConfig(loadedConfig);

      // Also save to localStorage for offline access
      localStorage.setItem(STORAGE_KEY, JSON.stringify(loadedConfig));
    } catch (err) {
      console.error('Failed to load model config:', err);

      // Try to load from localStorage as fallback
      const cached = localStorage.getItem(STORAGE_KEY);
      if (cached) {
        try {
          const cachedConfig = JSON.parse(cached) as FullModelConfig;
          setConfig(cachedConfig);
          setSavedConfig(cachedConfig);
        } catch {
          // Use defaults
        }
      }

      setError('Failed to load model configuration');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load config from server on mount
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const updateFastModel = useCallback((model: ModelConfig) => {
    setConfig((prev) => ({ ...prev, fastModel: model }));
  }, []);

  const updateSmartModel = useCallback((model: ModelConfig) => {
    setConfig((prev) => ({ ...prev, smartModel: model }));
  }, []);

  const updateOperationModel = useCallback(
    (operation: keyof OperationModelConfig, designation: ModelDesignation) => {
      setConfig((prev) => ({
        ...prev,
        operationModels: { ...prev.operationModels, [operation]: designation },
      }));
    },
    [],
  );

  const saveConfig = useCallback(async () => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await apiClient.postJson<{ success: boolean; config: FullModelConfig }>(
        '/api/admin/model-config',
        {
          fastModel: config.fastModel,
          smartModel: config.smartModel,
          operationModels: config.operationModels,
        },
      );

      if (response.success) {
        const newConfig = response.config;
        setConfig(newConfig);
        setSavedConfig(newConfig);

        // Update localStorage
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig));

        toast.success('Model configuration saved');
        return true;
      }

      throw new Error('Save failed');
    } catch (err) {
      console.error('Failed to save model config:', err);
      setError('Failed to save model configuration');
      toast.error('Failed to save model configuration');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [config]);

  const resetConfig = useCallback(() => {
    setConfig(savedConfig);
  }, [savedConfig]);

  const resetToDefaults = useCallback(async () => {
    setIsSaving(true);
    setError(null);

    try {
      await apiClient.deleteJson('/api/admin/model-config');
      await loadConfig();
      toast.success('Model configuration reset to defaults');
      return true;
    } catch (err) {
      console.error('Failed to reset model config:', err);
      setError('Failed to reset model configuration');
      toast.error('Failed to reset model configuration');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [loadConfig]);

  return {
    // Current config state
    config,
    savedConfig,

    // Status
    isLoading,
    isSaving,
    error,
    hasUnsavedChanges,

    // Config values (convenience accessors)
    fastModel: config.fastModel,
    smartModel: config.smartModel,
    operationModels: config.operationModels,

    // Actions
    updateFastModel,
    updateSmartModel,
    updateOperationModel,
    saveConfig,
    resetConfig,
    resetToDefaults,
    loadConfig,
  };
};

export default useModelConfig;
