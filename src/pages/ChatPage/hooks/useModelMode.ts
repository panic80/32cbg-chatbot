import { useEffect, useMemo } from 'react';
import { getModelDisplayName, LLM_MODELS } from '@/constants/models';
import { StorageKeys } from '@/constants/storage';
import { setLocalStorageItem, getLocalStorageItem } from '@/utils/storage';

// Default fallback models (used only if no config is saved)
const DEFAULT_MODELS = {
  fast: { provider: 'openai', model: 'gpt-4.1-mini' },
  smart: { provider: 'openai', model: 'gpt-5-mini' },
};

// Storage key for model config (must match useModelConfig.ts)
const MODEL_CONFIG_STORAGE_KEY = 'modelConfig';

interface ModelConfig {
  fastModel?: { provider: string; model: string };
  smartModel?: { provider: string; model: string };
}

/**
 * Get the configured fast/smart models from localStorage.
 * This reads the config saved by the config page.
 */
const getConfiguredModels = (): {
  fast: { provider: string; model: string };
  smart: { provider: string; model: string };
} => {
  try {
    const stored = getLocalStorageItem(MODEL_CONFIG_STORAGE_KEY);
    if (stored) {
      const config: ModelConfig = JSON.parse(stored);
      return {
        fast: config.fastModel || DEFAULT_MODELS.fast,
        smart: config.smartModel || DEFAULT_MODELS.smart,
      };
    }
  } catch (e) {
    console.warn('Failed to parse model config from localStorage:', e);
  }
  return DEFAULT_MODELS;
};

const getProviderForModel = (modelId: string): string => {
  const model = LLM_MODELS.find((item) => item.id === modelId);
  return model?.provider ?? 'openai';
};

export const useModelMode = (
  modelMode: 'fast' | 'smart',
  setCurrentModel: (model: string) => void,
) => {
  // Get configured models (re-read on each render to pick up config changes)
  const configuredModels = useMemo(() => getConfiguredModels(), []);

  useEffect(() => {
    // Get the model config for the selected mode
    const selectedConfig = modelMode === 'fast' ? configuredModels.fast : configuredModels.smart;
    const modelId = selectedConfig.model;
    const provider = selectedConfig.provider || getProviderForModel(modelId);

    setLocalStorageItem(StorageKeys.selectedModel, modelId);
    setLocalStorageItem(StorageKeys.selectedProvider, provider);
    setCurrentModel(getModelDisplayName(modelId));
  }, [modelMode, setCurrentModel, configuredModels]);
};
