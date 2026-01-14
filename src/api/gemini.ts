import { apiClient, ApiError } from '@/api/client';
import { parseApiResponse, ParsedChatResponse } from '../utils/chatUtils';
import { ChatError, ChatErrorType } from '../utils/chatErrors';

const MAX_RETRIES = 2;
const RETRY_DELAY = 1000; // 1 second

export type GeminiApiResponse = ParsedChatResponse & { fallback?: boolean };

const toError = (error: unknown): Error => {
  if (error instanceof Error) {
    return error;
  }
  return new Error(typeof error === 'string' ? error : JSON.stringify(error));
};

export const createPrompt = (
  message: string,
  isSimplified = false,
  instructions: string,
): string => {
  return `You are a helpful assistant for Canadian Forces Travel Instructions.

SPECIAL CALCULATION INSTRUCTIONS:
When a question involves travel entitlement calculations (e.g., "what are my entitlements", "total dollar amount", mentioning travel dates, mileage, R&Q, POMV, etc.), you MUST calculate the following components:

1. INCIDENTALS: Calculate daily incidentals for each day the member is away from home
2. MILEAGE: If POMV (Privately Owned Motor Vehicle) is authorized, calculate mileage costs based on distance provided
3. MEALS EN ROUTE: Calculate meal entitlements for travel days based on travel times:
   - If no specific times are provided, assume:
     * Departure from home: 10:00 AM on first travel day
     * Arrival home: 8:00 PM on last travel day
     * Departure from tasked location: after lunch timing on last day
   - Calculate meal entitlements based on these travel hours according to regulations

Provide a breakdown showing each component and the total dollar amount.

Here is the ONLY source material you can reference:
${instructions}

Question: ${message}


Please provide a response in this EXACT format:

Reference: <provide the section or chapter reference from the source>
Quote: <provide the exact quote that contains the answer>
${
  isSimplified
    ? 'Answer: <provide a concise answer in no more than two sentences>'
    : 'Answer: <provide a succinct one-sentence reply>\nReason: <provide a comprehensive explanation and justification drawing upon the source material>'
}`;
};

export const getGenerationConfig = () => ({
  temperature: 0.1,
  topP: 0.1,
  topK: 1,
  maxOutputTokens: 2048,
});

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const handleApiError = (error: unknown): ChatError => {
  if (error instanceof ChatError) {
    return error;
  }

  const normalised = toError(error);
  const message = normalised.message ?? '';

  if (message.includes('API key') || message.includes('authentication')) {
    return new ChatError(ChatErrorType.API_KEY, {
      message,
      details: 'Ensure GEMINI_API_KEY is configured on the server.',
    });
  }

  if (message.includes('quota') || message.includes('rate limit') || message.includes('429')) {
    return new ChatError(ChatErrorType.RATE_LIMIT, normalised);
  }

  if (
    message.includes('Network') ||
    message.includes('ECONNREFUSED') ||
    message.includes('fetch')
  ) {
    return new ChatError(ChatErrorType.NETWORK, normalised);
  }

  if (
    normalised instanceof SyntaxError ||
    message.includes('JSON') ||
    message.includes('Invalid response format')
  ) {
    return new ChatError(ChatErrorType.SERVICE, {
      message: 'Invalid API response format',
      details: message,
    });
  }

  return new ChatError(ChatErrorType.UNKNOWN, normalised);
};

export const callGeminiAPI = async (
  message: string,
  isSimplified: boolean,
  model: string,
  instructions: string,
  enableRetry = true,
): Promise<ParsedChatResponse> => {
  const promptText = createPrompt(message, isSimplified, instructions);
  const requestBody = {
    model,
    prompt: promptText,
    generationConfig: getGenerationConfig(),
  };

  const url = '/api/gemini/generateContent';
  const headers = {
    'Content-Type': 'application/json',
  } as const;

  let retries = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const response = await apiClient.request(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody),
        parseErrorResponse: true,
      });

      try {
        interface GeminiPart {
          text?: string;
        }
        interface GeminiCandidate {
          content?: { parts?: GeminiPart[] };
        }
        interface GeminiResponse {
          candidates?: GeminiCandidate[];
        }
        const data = (await response.json()) as GeminiResponse;

        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!text) {
          throw new Error('Invalid response format from Gemini API');
        }

        return parseApiResponse(text, isSimplified);
      } catch (parseError) {
        if (parseError instanceof SyntaxError) {
          throw new Error('Invalid JSON response from Gemini API');
        }
        throw parseError;
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 429) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      const normalised = toError(error);
      const message = normalised.message ?? '';

      if (
        enableRetry &&
        retries < MAX_RETRIES &&
        !message.includes('API key') &&
        !message.includes('Invalid response format') &&
        !(error instanceof ApiError && error.status === 429)
      ) {
        retries += 1;
        await delay(RETRY_DELAY * 2 ** (retries - 1));
        continue;
      }

      throw handleApiError(normalised);
    }
  }
};

export const callGeminiViaProxy = callGeminiAPI;
export const callGeminiViaSDK = callGeminiAPI;

export const getFallbackResponse = (isSimplified: boolean): GeminiApiResponse => ({
  text: isSimplified
    ? 'Unable to generate response. Please try again later.'
    : 'Unable to generate response. Please try again later. Our AI service may be experiencing temporary issues.',
  sources: [
    {
      id: 'fallback-system',
      reference: 'System',
      text: 'Fallback response when API is unavailable.',
    },
  ],
  isFormatted: true,
  fallback: true,
});

export const sendToGemini = async (
  message: string,
  isSimplified = false,
  model = 'gemini-2.5-flash',
  preloadedInstructions: string | null = null,
  useFallback = false,
): Promise<GeminiApiResponse> => {
  try {
    if (!preloadedInstructions) {
      throw new Error('Travel instructions not loaded');
    }

    return await callGeminiAPI(message, isSimplified, model, preloadedInstructions, true);
  } catch (error) {
    const normalised = toError(error);

    if (useFallback) {
      return getFallbackResponse(isSimplified);
    }

    if (!(normalised instanceof ChatError)) {
      const chatError = new ChatError(ChatErrorType.SERVICE, {
        message: 'The service is temporarily unavailable.',
        originalError: normalised,
      });
      throw chatError;
    }

    throw normalised;
  }
};
