import React, { useState, useEffect, useMemo } from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle, 
  Input, 
  Label, 
  AnimatedButton, 
  Badge 
} from '@/components/ui';
import {
  Loader2,
  CheckCircle,
  AlertCircle,
  Key,
  Search,
  ExternalLink,
  Star,
  Zap,
  Brain,
} from 'lucide-react';
import { apiClient } from '@/api/client';
import type { OpenRouterModel, OpenRouterModelsResponse } from '../types';

// Recommended models for RAG retrieval operations
const RECOMMENDED_RAG_MODELS = {
  fast: [
    {
      id: 'meta-llama/llama-3.1-8b-instruct',
      name: 'Llama 3.1 8B',
      reason: 'Best balance of speed and quality for query expansion & HyDE',
    },
    {
      id: 'mistralai/mistral-7b-instruct',
      name: 'Mistral 7B',
      reason: 'Fast, reliable, great for simple NLP tasks',
    },
    {
      id: 'qwen/qwen-2.5-7b-instruct',
      name: 'Qwen 2.5 7B',
      reason: 'Strong multilingual support, 128K context',
    },
  ],
  smart: [
    {
      id: 'meta-llama/llama-3.3-70b-instruct',
      name: 'Llama 3.3 70B',
      reason: 'Latest & best open-source model for RAG responses',
    },
    {
      id: 'qwen/qwen-2.5-72b-instruct',
      name: 'Qwen 2.5 72B',
      reason: 'Excellent reasoning, 128K context window',
    },
    {
      id: 'deepseek/deepseek-chat',
      name: 'DeepSeek Chat',
      reason: 'Strong reasoning at competitive pricing',
    },
  ],
};

interface OpenRouterTabProps {
  onModelsLoaded?: (models: OpenRouterModel[]) => void;
}

export const OpenRouterTab: React.FC<OpenRouterTabProps> = ({ onModelsLoaded }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [isConfigured, setIsConfigured] = useState(false);
  const [models, setModels] = useState<OpenRouterModel[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showOpenSourceOnly, setShowOpenSourceOnly] = useState(false);

  // Load models on mount
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const loadModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.getJson<OpenRouterModelsResponse>(
        '/api/admin/openrouter/models',
      );

      setModels(response.models || []);
      setIsConfigured(response.isConfigured);

      if (onModelsLoaded && response.models) {
        onModelsLoaded(response.models);
      }
    } catch (err: unknown) {
      setError('Failed to fetch OpenRouter models. Please try again.');
      console.error('OpenRouter models fetch error:', err instanceof Error ? err.message : err);
    } finally {
      setIsLoading(false);
    }
  }, [onModelsLoaded]);

  // Filter models based on search and open-source filter
  const filteredModels = useMemo(() => {
    let filtered = models;

    if (showOpenSourceOnly) {
      filtered = filtered.filter((m) => m.isOpenSource);
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (m) =>
          m.id.toLowerCase().includes(query) ||
          m.name.toLowerCase().includes(query) ||
          m.description?.toLowerCase().includes(query),
      );
    }

    return filtered;
  }, [models, searchQuery, showOpenSourceOnly]);

  const openSourceCount = useMemo(() => models.filter((m) => m.isOpenSource).length, [models]);

  return (
    <div className="space-y-4 animate-fade-up">
      {/* Configuration Status */}
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            OpenRouter Configuration
          </CardTitle>
          <CardDescription>
            Connect to OpenRouter for access to 200+ LLMs including open-source models.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <span className="ml-2 text-muted-foreground">Loading OpenRouter status...</span>
            </div>
          ) : isConfigured ? (
            <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <span className="text-green-700 dark:text-green-300">
                OpenRouter is configured ({models.length} models available, {openSourceCount}{' '}
                open-source)
              </span>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-amber-600" />
                <span className="text-amber-700 dark:text-amber-300">
                  OpenRouter API key not configured
                </span>
              </div>

              <div className="p-4 bg-muted rounded-lg space-y-3">
                <p className="text-sm font-medium">To enable OpenRouter:</p>
                <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
                  <li>
                    Get your API key from{' '}
                    <a
                      href="https://openrouter.ai/keys"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline inline-flex items-center gap-1"
                    >
                      openrouter.ai/keys
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </li>
                  <li>
                    Add{' '}
                    <code className="px-1 py-0.5 bg-background rounded">OPENROUTER_API_KEY</code> to
                    your server environment variables
                  </li>
                  <li>Restart the server to apply changes</li>
                </ol>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <span className="text-red-700 dark:text-red-300">{error}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recommended Models for RAG */}
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Star className="h-5 w-5 text-yellow-500" />
            Recommended Models for RAG
          </CardTitle>
          <CardDescription>
            Top-performing open-source models optimized for retrieval-augmented generation tasks.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Fast Models */}
          <div>
            <h4 className="flex items-center gap-2 font-medium mb-3">
              <Zap className="h-4 w-4 text-amber-500" />
              Fast Models
              <Badge variant="secondary" className="text-xs">
                For HyDE, Query Expansion
              </Badge>
            </h4>
            <div className="grid gap-2">
              {RECOMMENDED_RAG_MODELS.fast.map((model, idx) => (
                <div
                  key={model.id}
                  className="flex items-start gap-3 p-3 border rounded-lg bg-background/50 hover:border-amber-500/50 transition-colors"
                >
                  <div className="flex items-center justify-center w-6 h-6 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-600 text-sm font-medium shrink-0">
                    {idx + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium">{model.name}</div>
                    <code className="text-xs text-muted-foreground">{model.id}</code>
                    <p className="text-xs text-muted-foreground mt-1">{model.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Smart Models */}
          <div>
            <h4 className="flex items-center gap-2 font-medium mb-3">
              <Brain className="h-4 w-4 text-purple-500" />
              Smart Models
              <Badge variant="secondary" className="text-xs">
                For Response Generation
              </Badge>
            </h4>
            <div className="grid gap-2">
              {RECOMMENDED_RAG_MODELS.smart.map((model, idx) => (
                <div
                  key={model.id}
                  className="flex items-start gap-3 p-3 border rounded-lg bg-background/50 hover:border-purple-500/50 transition-colors"
                >
                  <div className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 text-sm font-medium shrink-0">
                    {idx + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium">{model.name}</div>
                    <code className="text-xs text-muted-foreground">{model.id}</code>
                    <p className="text-xs text-muted-foreground mt-1">{model.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm text-blue-700 dark:text-blue-300">
              <strong>Tip:</strong> Use a Fast model for HyDE/query expansion and a Smart model for
              response generation. Configure this in the <strong>RAG Config</strong> tab.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Models List */}
      {models.length > 0 && (
        <Card className="glass border-border/50">
          <CardHeader>
            <CardTitle>Available Models ({filteredModels.length})</CardTitle>
            <CardDescription>
              Browse and search available models. Use these model IDs when configuring Fast/Smart
              models.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Search and Filter */}
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search models..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="open-source-toggle" className="text-sm cursor-pointer">
                  Open-source only
                </Label>
                <input
                  id="open-source-toggle"
                  type="checkbox"
                  checked={showOpenSourceOnly}
                  onChange={(e) => setShowOpenSourceOnly(e.target.checked)}
                  className="h-4 w-4 rounded border-border"
                />
              </div>
            </div>

            {/* Models Grid */}
            <div className="max-h-96 overflow-y-auto space-y-2 pr-2">
              {filteredModels.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No models match your search criteria
                </div>
              ) : (
                filteredModels.slice(0, 50).map((model) => (
                  <div
                    key={model.id}
                    className="p-3 border rounded-lg hover:border-primary/50 transition-colors bg-background/50"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-medium truncate">{model.name}</span>
                          {model.isOpenSource && (
                            <Badge variant="secondary" className="text-xs">
                              Open Source
                            </Badge>
                          )}
                        </div>
                        <code className="text-xs text-muted-foreground block truncate mt-1">
                          {model.id}
                        </code>
                        {model.description && (
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {model.description}
                          </p>
                        )}
                      </div>
                      <div className="text-right text-xs text-muted-foreground shrink-0">
                        <div>{model.contextLength?.toLocaleString() || 'N/A'} ctx</div>
                        {model.pricing && (
                          <div className="text-green-600 dark:text-green-400">
                            ${parseFloat(model.pricing.prompt).toFixed(4)}/1K
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
              {filteredModels.length > 50 && (
                <div className="text-center py-2 text-sm text-muted-foreground">
                  Showing first 50 of {filteredModels.length} models. Use search to find specific
                  models.
                </div>
              )}
            </div>

            {/* Refresh Button */}
            <div className="flex justify-end">
              <AnimatedButton
                variant="outline"
                size="sm"
                onClick={loadModels}
                disabled={isLoading}
                ripple
              >
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Refresh Models
              </AnimatedButton>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default OpenRouterTab;
