export interface LLMModel {
  id: string;
  name: string;
  provider: 'openai' | 'google' | 'anthropic' | 'openrouter';
  description?: string;
  tier?: 'fast' | 'smart' | 'balanced'; // Suggested tier for fast/smart selection
}

export const LLM_MODELS: LLMModel[] = [
  // OpenAI Models - Latest 2025 API Identifiers (verified via web search)
  {
    id: 'o3',
    name: 'O3',
    provider: 'openai',
    description: 'Most powerful reasoning model for complex logical tasks',
  },
  {
    id: 'o4-mini',
    name: 'O4 Mini',
    provider: 'openai',
    description: 'Fast, cost-efficient reasoning model optimized for math and coding',
  },
  {
    id: 'gpt-4.1',
    name: 'GPT-4.1',
    provider: 'openai',
    description: 'Latest GPT-4.1 with major improvements in coding and instruction following',
  },
  {
    id: 'gpt-4.1-mini',
    name: 'GPT-4.1 Mini',
    provider: 'openai',
    description: 'Lightweight version of GPT-4.1 for everyday coding needs',
  },
  {
    id: 'gpt-4.1-nano',
    name: 'GPT-4.1 Nano',
    provider: 'openai',
    description: 'Ultra-lightweight GPT-4.1 model for fast, cost-efficient responses',
  },
  {
    id: 'gpt-5-mini',
    name: 'GPT-5 Mini',
    provider: 'openai',
    description: 'Next-generation efficient model with improved reasoning and cost-effectiveness',
  },
  {
    id: 'gpt-5-nano',
    name: 'GPT-5 Nano',
    provider: 'openai',
    description:
      'Ultra-lightweight GPT-5 family model for rapid responses with excellent cost-efficiency',
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: 'openai',
    description: 'Flagship multimodal model with vision and text capabilities',
  },
  {
    id: 'gpt-4o-mini',
    name: 'GPT-4o Mini',
    provider: 'openai',
    description: 'Efficient and cost-effective GPT-4 model',
  },

  // Google Models - Current 2025 API Identifiers
  {
    id: 'gemini-3-pro-preview',
    name: 'Gemini 3 Pro',
    provider: 'google',
    description: 'Best model for multimodal understanding and agentic/vibe-coding',
  },
  {
    id: 'gemini-3-flash-preview',
    name: 'Gemini 3 Flash Preview',
    provider: 'google',
    description: 'Frontier-class performance at a fraction of the cost',
  },
  {
    id: 'gemini-2.5-pro',
    name: 'Gemini 2.5 Pro',
    provider: 'google',
    description: 'Most capable and intelligent Gemini model',
  },
  {
    id: 'gemini-2.5-flash',
    name: 'Gemini 2.5 Flash',
    provider: 'google',
    description: 'Best price-performance model with thinking capabilities',
  },
  {
    id: 'gemini-2.5-flash-lite',
    name: 'Gemini 2.5 Flash Lite',
    provider: 'google',
    description: 'Lightweight and cost-effective',
  },

  // Anthropic Models - Current 2025 API Identifiers
  {
    id: 'claude-sonnet-4-20250514',
    name: 'Claude Sonnet 4',
    provider: 'anthropic',
    description: 'Next-gen model excelling in coding and advanced reasoning',
  },
];

/**
 * Get the display name for a model ID
 * @param modelId - The model ID stored in localStorage
 * @returns The display name for the model, or the ID itself if not found
 */
export const getModelDisplayName = (modelId: string | null): string => {
  if (!modelId) {
    return 'GPT-4o Mini'; // Default fallback
  }

  const model = LLM_MODELS.find((m) => m.id === modelId);
  return model ? model.name : modelId;
};

/**
 * Get the default model ID
 */
export const DEFAULT_MODEL_ID = 'gpt-5-mini';
