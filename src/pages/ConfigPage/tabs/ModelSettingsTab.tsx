import React from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AnimatedButton } from '@/components/ui/animated-button';
import { CheckCircle } from 'lucide-react';
import type { LLMModel } from '@/constants/models';
import type { ModelProvider } from '../types';

interface ModelSettingsTabProps {
  models: LLMModel[];
  selectedModel: string;
  tempSelectedModel: string;
  tempSelectedProvider: ModelProvider;
  hasUnsavedChanges: boolean;
  onProviderChange: (provider: ModelProvider) => void;
  onModelChange: (modelId: string) => void;
  onSave: () => void;
  onReset: () => void;
}

export const ModelSettingsTab: React.FC<ModelSettingsTabProps> = ({
  models,
  selectedModel,
  tempSelectedModel,
  tempSelectedProvider,
  hasUnsavedChanges,
  onProviderChange,
  onModelChange,
  onSave,
  onReset,
}) => {
  const providerModels = React.useMemo(
    () => models.filter((model) => model.provider === tempSelectedProvider),
    [models, tempSelectedProvider],
  );

  return (
    <div className="space-y-4 animate-fade-up">
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle>LLM Model Selection</CardTitle>
          <CardDescription>
            Choose your preferred AI model for the chat assistant. Different models offer varying
            capabilities and performance.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Tabs
            value={tempSelectedProvider}
            onValueChange={(value) => onProviderChange(value as ModelProvider)}
          >
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="openai">OpenAI</TabsTrigger>
              <TabsTrigger value="google">Google</TabsTrigger>
              <TabsTrigger value="anthropic">Anthropic</TabsTrigger>
            </TabsList>

            <div className="mt-4 space-y-2">
              {providerModels.map((model) => {
                const isSelected = tempSelectedModel === model.id;
                return (
                  <div
                    key={model.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      isSelected
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:border-primary/50'
                    }`}
                    onClick={() => onModelChange(model.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">{model.name}</h4>
                        {model.description && (
                          <p className="text-sm text-muted-foreground mt-1">{model.description}</p>
                        )}
                      </div>
                      {isSelected && <CheckCircle className="h-5 w-5 text-primary" />}
                    </div>
                  </div>
                );
              })}
            </div>
          </Tabs>

          {hasUnsavedChanges && (
            <div className="flex gap-2 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
              <div className="flex-1">
                <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  You have unsaved changes
                </p>
                <p className="text-xs text-yellow-600 dark:text-yellow-300">
                  Save your changes to apply the new model selection.
                </p>
              </div>
              <div className="flex gap-2">
                <AnimatedButton variant="outline" size="sm" onClick={onReset} ripple>
                  Reset
                </AnimatedButton>
                <AnimatedButton size="sm" onClick={onSave} ripple>
                  Save Changes
                </AnimatedButton>
              </div>
            </div>
          )}

          <div className="mt-6 p-4 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">
              <strong>Active Model:</strong>{' '}
              {models.find((model) => model.id === selectedModel)?.name || 'None selected'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Model ID: <code className="px-1 py-0.5 bg-background rounded">{selectedModel}</code>
            </p>
            {hasUnsavedChanges && (
              <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
                <strong>Selected:</strong>{' '}
                {models.find((model) => model.id === tempSelectedModel)?.name} (not saved)
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
