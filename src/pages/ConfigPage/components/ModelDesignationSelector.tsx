import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Zap, Brain, Settings2, Star } from 'lucide-react';
import type { LLMModel } from '@/constants/models';
import type {
  ModelConfig,
  OperationModelConfig,
  ModelDesignation,
  ModelProvider,
  OpenRouterModel,
} from '../types';

// Recommended OpenRouter models that should always be available
const RECOMMENDED_OPENROUTER_MODELS = [
  // Fast models
  { id: 'meta-llama/llama-3.1-8b-instruct', name: 'Llama 3.1 8B (Fast)', recommended: 'fast' },
  { id: 'mistralai/mistral-7b-instruct', name: 'Mistral 7B (Fast)', recommended: 'fast' },
  { id: 'qwen/qwen-2.5-7b-instruct', name: 'Qwen 2.5 7B (Fast)', recommended: 'fast' },
  // Smart models
  { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B (Smart)', recommended: 'smart' },
  { id: 'qwen/qwen-2.5-72b-instruct', name: 'Qwen 2.5 72B (Smart)', recommended: 'smart' },
  { id: 'deepseek/deepseek-chat', name: 'DeepSeek Chat (Smart)', recommended: 'smart' },
];

interface ModelDesignationSelectorProps {
  // All available models from standard providers
  standardModels: LLMModel[];
  // OpenRouter models (if available)
  openRouterModels?: OpenRouterModel[];
  // Current configuration
  fastModel: ModelConfig;
  smartModel: ModelConfig;
  operationModels: OperationModelConfig;
  // Callbacks
  onFastModelChange: (config: ModelConfig) => void;
  onSmartModelChange: (config: ModelConfig) => void;
  onOperationModelChange: (
    operation: keyof OperationModelConfig,
    designation: ModelDesignation,
  ) => void;
}

type CombinedModel = {
  id: string;
  name: string;
  provider: ModelProvider;
  description?: string;
  isRecommended?: boolean;
};

export const ModelDesignationSelector: React.FC<ModelDesignationSelectorProps> = ({
  standardModels,
  openRouterModels = [],
  fastModel,
  smartModel,
  operationModels,
  onFastModelChange,
  onSmartModelChange,
  onOperationModelChange,
}) => {
  // Combine all models into a single list with provider grouping
  const allModels = useMemo((): CombinedModel[] => {
    const combined: CombinedModel[] = [
      ...standardModels.map((m) => ({
        id: m.id,
        name: m.name,
        provider: m.provider as ModelProvider,
        description: m.description,
      })),
    ];

    // Always include recommended OpenRouter models first
    const recommendedIds = new Set(RECOMMENDED_OPENROUTER_MODELS.map((m) => m.id));
    const recommendedModels = RECOMMENDED_OPENROUTER_MODELS.map((m) => ({
      id: m.id,
      name: m.name,
      provider: 'openrouter' as ModelProvider,
      description:
        m.recommended === 'fast'
          ? 'Recommended for fast operations'
          : 'Recommended for smart operations',
      isRecommended: true,
    }));

    // Add other OpenRouter models (excluding already recommended ones)
    const otherOpenRouterModels = openRouterModels
      .filter((m) => m.isOpenSource && !recommendedIds.has(m.id))
      .slice(0, 15)
      .map((m) => ({
        id: m.id,
        name: m.name,
        provider: 'openrouter' as ModelProvider,
        description: m.description,
      }));

    return [...combined, ...recommendedModels, ...otherOpenRouterModels];
  }, [standardModels, openRouterModels]);

  // Group models by provider
  const modelsByProvider = useMemo(() => {
    const grouped: Record<ModelProvider, CombinedModel[]> = {
      openai: [],
      google: [],
      anthropic: [],
      openrouter: [],
    };

    allModels.forEach((model) => {
      if (grouped[model.provider]) {
        grouped[model.provider].push(model);
      }
    });

    return grouped;
  }, [allModels]);

  const handleModelSelect = (type: 'fast' | 'smart', fullId: string) => {
    const [provider, ...modelParts] = fullId.split(':');
    const model = modelParts.join(':');
    const config: ModelConfig = { provider: provider as ModelProvider, model };

    if (type === 'fast') {
      onFastModelChange(config);
    } else {
      onSmartModelChange(config);
    }
  };

  const getFullModelId = (config: ModelConfig) => `${config.provider}:${config.model}`;

  const providerLabels: Record<ModelProvider, string> = {
    openai: 'OpenAI',
    google: 'Google',
    anthropic: 'Anthropic',
    openrouter: 'OpenRouter',
  };

  const operations = [
    {
      key: 'responseGeneration' as const,
      label: 'Response Generation',
      desc: 'Main chat answer from RAG',
    },
    {
      key: 'hydeExpansion' as const,
      label: 'HyDE Query Expansion',
      desc: 'Hypothetical document generation',
    },
    {
      key: 'queryRewriting' as const,
      label: 'Query Rewriting',
      desc: 'Reformulating user queries',
    },
    {
      key: 'followUpGeneration' as const,
      label: 'Follow-up Questions',
      desc: 'Suggested next questions',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Fast/Smart Model Selection */}
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings2 className="h-5 w-5" />
            Model Designation
          </CardTitle>
          <CardDescription>
            Designate one model as "Fast" (cost-efficient, quick) and one as "Smart" (powerful,
            accurate). These can be from any provider including OpenRouter.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Fast Model */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-amber-500" />
              Fast Model
            </Label>
            <Select
              value={getFullModelId(fastModel)}
              onValueChange={(v) => handleModelSelect('fast', v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select fast model" />
              </SelectTrigger>
              <SelectContent>
                {(Object.entries(modelsByProvider) as [ModelProvider, CombinedModel[]][])
                  .filter(([, models]) => models.length > 0)
                  .map(([provider, models]) => (
                    <React.Fragment key={provider}>
                      <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase bg-muted/50">
                        {providerLabels[provider]}
                      </div>
                      {models.map((m) => (
                        <SelectItem key={`${provider}:${m.id}`} value={`${provider}:${m.id}`}>
                          <span className="flex items-center gap-1.5 truncate">
                            {m.isRecommended && (
                              <Star className="h-3 w-3 text-yellow-500 shrink-0" />
                            )}
                            {m.name}
                          </span>
                        </SelectItem>
                      ))}
                    </React.Fragment>
                  ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Used for quick operations like query expansion and follow-up generation.
            </p>
          </div>

          {/* Smart Model */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-purple-500" />
              Smart Model
            </Label>
            <Select
              value={getFullModelId(smartModel)}
              onValueChange={(v) => handleModelSelect('smart', v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select smart model" />
              </SelectTrigger>
              <SelectContent>
                {(Object.entries(modelsByProvider) as [ModelProvider, CombinedModel[]][])
                  .filter(([, models]) => models.length > 0)
                  .map(([provider, models]) => (
                    <React.Fragment key={provider}>
                      <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase bg-muted/50">
                        {providerLabels[provider]}
                      </div>
                      {models.map((m) => (
                        <SelectItem key={`${provider}:${m.id}`} value={`${provider}:${m.id}`}>
                          <span className="flex items-center gap-1.5 truncate">
                            {m.isRecommended && (
                              <Star className="h-3 w-3 text-yellow-500 shrink-0" />
                            )}
                            {m.name}
                          </span>
                        </SelectItem>
                      ))}
                    </React.Fragment>
                  ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Used for main response generation requiring higher accuracy.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Operation Configuration */}
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle>Operation Model Assignment</CardTitle>
          <CardDescription>
            Configure which model type (Fast or Smart) is used for each RAG operation.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {operations.map(({ key, label, desc }) => (
            <div
              key={key}
              className="flex items-center justify-between p-3 border rounded-lg bg-background/50"
            >
              <div className="flex-1 min-w-0 mr-4">
                <div className="font-medium">{label}</div>
                <div className="text-xs text-muted-foreground">{desc}</div>
              </div>
              <Select
                value={operationModels[key]}
                onValueChange={(v) => onOperationModelChange(key, v as ModelDesignation)}
              >
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fast">
                    <span className="flex items-center gap-2">
                      <Zap className="h-3 w-3 text-amber-500" />
                      Fast
                    </span>
                  </SelectItem>
                  <SelectItem value="smart">
                    <span className="flex items-center gap-2">
                      <Brain className="h-3 w-3 text-purple-500" />
                      Smart
                    </span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelDesignationSelector;
