import { apiClient, ApiError } from '@/api/client';
import { StorageKeys } from '@/constants/storage';
import { getLocalStorageItem } from '@/utils/storage';
import { FollowUpQuestion, Source } from '@/types/chat';
import { FOLLOW_UP_CATEGORIES, DEFAULT_CONFIDENCE, FALLBACK_QUESTIONS } from '@/constants';

interface FollowUpGenerationParams {
  userQuestion: string;
  aiResponse: string;
  sources?: Source[];
  conversationHistory?: Array<{ role: 'user' | 'assistant'; content: string }>;
}

interface FollowUpResponse {
  followUpQuestions: FollowUpQuestion[];
}

interface RawFollowUpQuestion {
  question?: string;
  category?: FollowUpQuestion['category'] | string;
  confidence?: number;
}

/**
 * Generate contextual follow-up questions using the dedicated follow-up endpoint
 */
export const generateFollowUpQuestions = async (
  params: FollowUpGenerationParams,
): Promise<FollowUpQuestion[]> => {
  const { userQuestion, aiResponse, sources = [], conversationHistory = [] } = params;

  try {
    // Call the dedicated follow-up endpoint
    const data = await apiClient.postJson<FollowUpResponse>(
      '/api/v2/followup',
      {
        userQuestion,
        aiResponse,
        sources,
        conversationHistory,
        model: getLocalStorageItem(StorageKeys.selectedModel) || 'gpt-4',
        provider: getLocalStorageItem(StorageKeys.selectedProvider) || 'openai',
      },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      },
    );

    return data?.followUpQuestions || [];
  } catch (error) {
    if (error instanceof ApiError) {
      console.error(
        'Error generating follow-up questions:',
        error.status,
        error.statusText,
        error.data,
      );
    } else {
      console.error('Error generating follow-up questions:', error);
    }
    // Return fallback questions based on context analysis
    return generateFallbackQuestions(userQuestion, aiResponse, sources);
  }
};

/**
 * Create a specialized prompt for generating follow-up questions
 */
const createFollowUpPrompt = (
  userQuestion: string,
  aiResponse: string,
  sources: Source[],
  conversationHistory: Array<{ role: 'user' | 'assistant'; content: string }>,
): string => {
  const sourceContext =
    sources.length > 0
      ? `\n\nSources referenced: ${sources.map((s) => s.reference || 'Document').join(', ')}`
      : '';

  const historyContext =
    conversationHistory.length > 0
      ? `\n\nRecent conversation context:\n${conversationHistory
          .slice(-4)
          .map((h) => `${h.role}: ${h.content.substring(0, 100)}...`)
          .join('\n')}`
      : '';

  return `You are a helpful assistant specialized in generating contextual follow-up questions. Based on the conversation below, generate 2-3 relevant follow-up questions that would help the user continue learning or get more specific information.

User's Question: "${userQuestion}"

AI Response: "${aiResponse}"${sourceContext}${historyContext}

Please generate follow-up questions in the following JSON format:
{
  "questions": [
    {
      "question": "Specific follow-up question text",
      "category": "${FOLLOW_UP_CATEGORIES.CLARIFICATION}|${FOLLOW_UP_CATEGORIES.RELATED}|${FOLLOW_UP_CATEGORIES.PRACTICAL}|${FOLLOW_UP_CATEGORIES.EXPLORE}",
      "confidence": 0.9
    }
  ]
}

Focus on:
- Clarification questions for unclear aspects
- Related topics that might be relevant
- Practical application questions
- Questions that explore deeper into the subject

Generate questions that are:
- Specific and actionable
- Relevant to the context
- Naturally conversational
- Helpful for learning more

Return only the JSON response.`;
};

/**
 * Parse AI response to extract follow-up questions
 */
const parseFollowUpQuestions = (response: string): FollowUpQuestion[] => {
  try {
    // Try to extract JSON from the response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No JSON found in response');
    }

    const parsed = JSON.parse(jsonMatch[0]);
    const questions: RawFollowUpQuestion[] = Array.isArray(parsed.questions)
      ? parsed.questions
      : [];

    return questions
      .map((q: RawFollowUpQuestion, index: number) => {
        // Validate category against known types, default to RELATED
        let category: FollowUpQuestion['category'] = FOLLOW_UP_CATEGORIES.RELATED;
        const validCategories = Object.values(FOLLOW_UP_CATEGORIES) as string[];
        if (q.category && validCategories.includes(q.category)) {
          category = q.category as FollowUpQuestion['category'];
        }

        return {
          id: `followup-${Date.now()}-${index}`,
          question: q.question || '',
          category: category,
          confidence: q.confidence || DEFAULT_CONFIDENCE.MEDIUM,
        };
      })
      .filter((q: FollowUpQuestion) => q.question.trim().length > 0);
  } catch (error) {
    console.error('Error parsing follow-up questions:', error);
    // Try to extract questions from plain text as fallback
    return extractQuestionsFromText(response);
  }
};

/**
 * Extract questions from plain text as fallback
 */
const extractQuestionsFromText = (text: string): FollowUpQuestion[] => {
  const lines = text.split('\n');
  const questions: FollowUpQuestion[] = [];

  lines.forEach((line, index) => {
    // Look for lines that end with ? or start with question indicators
    const trimmed = line.trim();
    if (trimmed.endsWith('?') && trimmed.length > 10) {
      // Remove common prefixes
      const cleanQuestion = trimmed
        .replace(/^\d+\.\s*/, '')
        .replace(/^[-*]\s*/, '')
        .replace(/^Q:\s*/i, '')
        .trim();

      if (cleanQuestion.length > 10) {
        questions.push({
          id: `followup-text-${Date.now()}-${index}`,
          question: cleanQuestion,
          category: FOLLOW_UP_CATEGORIES.RELATED,
          confidence: DEFAULT_CONFIDENCE.LOW,
        });
      }
    }
  });

  return questions.slice(0, 3); // Limit to 3 questions
};

/**
 * Generate fallback questions when AI generation fails
 */
const generateFallbackQuestions = (
  userQuestion: string,
  aiResponse: string,
  sources: Source[],
): FollowUpQuestion[] => {
  const fallbackQuestions: FollowUpQuestion[] = [];

  // Analyze the content to generate contextual questions
  const questionLower = userQuestion.toLowerCase();
  const responseLower = aiResponse.toLowerCase();

  // Policy/travel specific fallbacks
  if (questionLower.includes('travel') || responseLower.includes('travel')) {
    fallbackQuestions.push({
      id: 'fallback-travel-1',
      question: FALLBACK_QUESTIONS.TRAVEL,
      category: FOLLOW_UP_CATEGORIES.PRACTICAL,
      confidence: DEFAULT_CONFIDENCE.FALLBACK,
    });
  }

  if (questionLower.includes('claim') || responseLower.includes('claim')) {
    fallbackQuestions.push({
      id: 'fallback-claim-1',
      question: FALLBACK_QUESTIONS.CLAIM,
      category: FOLLOW_UP_CATEGORIES.PRACTICAL,
      confidence: DEFAULT_CONFIDENCE.FALLBACK,
    });
  }

  if (questionLower.includes('allowance') || responseLower.includes('allowance')) {
    fallbackQuestions.push({
      id: 'fallback-allowance-1',
      question: FALLBACK_QUESTIONS.ALLOWANCE,
      category: FOLLOW_UP_CATEGORIES.CLARIFICATION,
      confidence: DEFAULT_CONFIDENCE.FALLBACK,
    });
  }

  // Generic fallbacks
  if (fallbackQuestions.length === 0) {
    fallbackQuestions.push(
      {
        id: 'fallback-generic-1',
        question: FALLBACK_QUESTIONS.GENERIC_EXAMPLES,
        category: FOLLOW_UP_CATEGORIES.CLARIFICATION,
        confidence: DEFAULT_CONFIDENCE.MINIMAL,
      },
      {
        id: 'fallback-generic-2',
        question: FALLBACK_QUESTIONS.GENERIC_NEXT_STEPS,
        category: FOLLOW_UP_CATEGORIES.PRACTICAL,
        confidence: DEFAULT_CONFIDENCE.MINIMAL,
      },
    );
  }

  return fallbackQuestions.slice(0, 2); // Limit fallback questions
};

export default {
  generateFollowUpQuestions,
};
