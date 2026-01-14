import { useState, useEffect, useCallback } from 'react';
import { StorageKeys } from '@/constants/storage';
import { getLocalStorageItem, setLocalStorageItem } from '@/utils/storage';
import type { LLMModel } from '@/constants/models';
import type { ModelProvider } from '../types';

const isModelProvider = (value: string | null): value is ModelProvider => {
  return value === 'openai' || value === 'google' || value === 'anthropic';
};

export interface ModelPreferencesHook {
  selectedModel: string;
  selectedProvider: ModelProvider;
  tempSelectedModel: string;
  tempSelectedProvider: ModelProvider;
  hasUnsavedChanges: boolean;
  selectModel: (modelId: string) => void;
  selectProvider: (provider: ModelProvider) => void;
  savePreferences: () => LLMModel | null;
  resetPreferences: () => void;
}

export const useModelPreferences = (
  models: LLMModel[],
  defaultModelId: string,
  fallbackProvider: ModelProvider,
): ModelPreferencesHook => {
  const defaultProvider =
    (models.find((model) => model.id === defaultModelId)?.provider as ModelProvider) ||
    fallbackProvider;

  const [selectedModel, setSelectedModel] = useState<string>(defaultModelId);
  const [selectedProvider, setSelectedProvider] = useState<ModelProvider>(defaultProvider);
  const [tempSelectedModel, setTempSelectedModel] = useState<string>(defaultModelId);
  const [tempSelectedProvider, setTempSelectedProvider] = useState<ModelProvider>(defaultProvider);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  useEffect(() => {
    const savedModel = getLocalStorageItem(StorageKeys.selectedModel);
    const savedProvider = getLocalStorageItem(StorageKeys.selectedProvider);

    const initialModel = models.find((model) => model.id === savedModel)?.id || defaultModelId;
    const initialProvider = isModelProvider(savedProvider)
      ? savedProvider
      : (models.find((model) => model.id === initialModel)?.provider as ModelProvider) ||
        defaultProvider;

    setSelectedModel(initialModel);
    setSelectedProvider(initialProvider);
    setTempSelectedModel(initialModel);
    setTempSelectedProvider(initialProvider);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultModelId, defaultProvider, models]);

  const selectProvider = useCallback(
    (provider: ModelProvider) => {
      setTempSelectedProvider(provider);
      setHasUnsavedChanges(provider !== selectedProvider || tempSelectedModel !== selectedModel);
    },
    [selectedModel, selectedProvider, tempSelectedModel],
  );

  const selectModel = useCallback(
    (modelId: string) => {
      const model = models.find((item) => item.id === modelId);
      if (!model) return;

      const provider = model.provider as ModelProvider;
      setTempSelectedModel(modelId);
      setTempSelectedProvider(provider);
      setHasUnsavedChanges(modelId !== selectedModel || provider !== selectedProvider);
    },
    [models, selectedModel, selectedProvider],
  );

  const savePreferences = useCallback(() => {
    const model = models.find((item) => item.id === tempSelectedModel);
    if (!model) {
      return null;
    }

    const provider = tempSelectedProvider;

    setSelectedModel(tempSelectedModel);
    setSelectedProvider(provider);
    setHasUnsavedChanges(false);

    setLocalStorageItem(StorageKeys.selectedModel, tempSelectedModel);
    setLocalStorageItem(StorageKeys.selectedProvider, provider);

    return model;
  }, [models, tempSelectedModel, tempSelectedProvider]);

  const resetPreferences = useCallback(() => {
    setTempSelectedModel(selectedModel);
    setTempSelectedProvider(selectedProvider);
    setHasUnsavedChanges(false);
  }, [selectedModel, selectedProvider]);

  return {
    selectedModel,
    selectedProvider,
    tempSelectedModel,
    tempSelectedProvider,
    hasUnsavedChanges,
    selectModel,
    selectProvider,
    savePreferences,
    resetPreferences,
  };
};
