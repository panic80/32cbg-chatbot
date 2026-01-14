import axios, { AxiosInstance } from 'axios';
import type { Request, Response } from 'express';
import { AiClients } from '../services/aiClients.js';

interface SupportControllerConfig extends Partial<AiClients> {
  processContent: (html: string) => string;
  cache: import('../services/cache.js').CacheService | null;
  config: {
    maxRetries: number;
    canadaCaUrl: string;
    requestTimeout: number;
    retryDelay: number;
  };
  httpClient?: AxiosInstance;
}

interface FollowUpQuestion {
  id: string;
  question: string;
  category: string;
  confidence: number;
}

const FALLBACK_FOLLOW_UPS: Omit<FollowUpQuestion, 'id'>[] = [
  {
    question: 'Can you provide more specific examples?',
    category: 'clarification',
    confidence: 0.5,
  },
  {
    question: 'What are the next steps I should take?',
    category: 'practical',
    confidence: 0.5,
  },
];

const parseQuestions = (text: string): FollowUpQuestion[] => {
  try {
    const jsonMatch = text.match(/[\[\s\S]*\]/);
    if (!jsonMatch) {
      return [];
    }
    const questions = JSON.parse(jsonMatch[0]);
    return (
      questions as Array<{ question?: string; category?: string; confidence?: number } | string>
    )
      .map((q, idx: number) => ({
        id: `followup-${Date.now()}-${idx}`,
        question: typeof q === 'string' ? q : (q?.question ?? ''),
        category: typeof q === 'string' ? 'related' : q?.category || 'related',
        confidence: typeof q === 'string' ? 0.7 : q?.confidence || 0.7,
      }))
      .filter((q: FollowUpQuestion) => q.question.trim().length > 0);
  } catch (error) {
    return [];
  }
};

export const createSupportController = ({
  geminiClient,
  openaiClient,
  anthropicClient,
  processContent,
  cache,
  config,
  httpClient = axios,
}: SupportControllerConfig) => {
  const handleFollowUp = async (req: Request, res: Response) => {
    const { userQuestion, aiResponse, model = 'gemini-2.0-flash', provider = 'google' } = req.body;

    try {
      const prompt = `Based on this conversation, generate 2-3 relevant follow-up questions:\n\nUser Question: "${userQuestion}"\nAI Response: "${aiResponse}"\n\nGenerate follow-up questions that would help the user learn more or get specific information. Return as a JSON array of questions.`;

      let followUpQuestions: FollowUpQuestion[] = [];

      switch (provider) {
        case 'google':
          if (geminiClient) {
            const modelInstance = geminiClient.getGenerativeModel({ model });
            const result = await modelInstance.generateContent(prompt);
            const text = await result.response.text();
            followUpQuestions = parseQuestions(text);
          }
          break;
        case 'openai':
          if (openaiClient) {
            const completion = await openaiClient.chat.completions.create({
              model,
              messages: [{ role: 'user', content: prompt }],
              temperature: 0.7,
            });
            const text = completion.choices[0].message.content || '';
            followUpQuestions = parseQuestions(text);
          }
          break;
        case 'anthropic':
          if (anthropicClient) {
            const response = await anthropicClient.messages.create({
              model,
              max_tokens: 4096,
              messages: [{ role: 'user', content: prompt }],
            });
            const firstContent = response.content?.[0];
            const text =
              firstContent && 'text' in firstContent ? (firstContent.text as string) : '';
            followUpQuestions = parseQuestions(text);
          }
          break;
        default:
          break;
      }

      if (followUpQuestions.length === 0) {
        followUpQuestions = FALLBACK_FOLLOW_UPS.map((item, idx) => ({
          id: `followup-${Date.now()}-${idx}`,
          ...item,
        }));
      }

      return res.json({ followUpQuestions });
    } catch (error) {
      return res.json({
        followUpQuestions: FALLBACK_FOLLOW_UPS.map((item, idx) => ({
          id: `followup-${Date.now()}-${idx}`,
          ...item,
        })),
      });
    }
  };

  const handleTravelInstructions = async (req: Request, res: Response) => {
    try {
      const startTime = Date.now();
      const ifNoneMatch = req.headers['if-none-match'];

      if (cache) {
        interface CachedInstructions {
          content: string;
          timestamp: number;
          lastModified?: string;
          etag: string;
        }
        const cachedData = await cache.get<CachedInstructions>('travel-instructions');
        if (cachedData && cachedData.content && cachedData.etag) {
          if (ifNoneMatch && ifNoneMatch === cachedData.etag) {
            return res.status(304).send();
          }

          res.setHeader('Cache-Control', 'public, max-age=3600');
          res.setHeader('ETag', cachedData.etag);
          if (cachedData.lastModified) {
            res.setHeader('Last-Modified', cachedData.lastModified);
          }

          return res.json({
            content: cachedData.content,
            fresh: false,
            cacheAge: Date.now() - cachedData.timestamp,
            timestamp: new Date(cachedData.timestamp).toISOString(),
          });
        }
      }

      let response;
      let lastError;

      for (let attempt = 1; attempt <= config.maxRetries; attempt += 1) {
        try {
          response = await httpClient.get(config.canadaCaUrl, {
            timeout: config.requestTimeout,
            headers: {
              'User-Agent': 'Mozilla/5.0 (compatible; CFTravelBot/1.0)',
              Accept: 'text/html,application/xhtml+xml',
              'Accept-Language': 'en-CA,en;q=0.9',
              'Cache-Control': 'no-cache',
            },
            validateStatus: (status) => status < 500,
          });

          if (response.status === 200) {
            break;
          }

          if (response.status >= 400 && response.status < 500) {
            throw new Error(`Canada.ca returned status ${response.status}`);
          }
        } catch (error) {
          lastError = error;
          if (attempt < config.maxRetries) {
            await new Promise((resolve) => setTimeout(resolve, config.retryDelay * attempt));
          }
        }
      }

      if (!response || response.status !== 200) {
        throw lastError || new Error('Failed to fetch travel instructions after all retries');
      }

      const content = processContent(response.data);
      const etag = `"${Buffer.from(content).toString('base64').substring(0, 27)}"`;

      if (cache) {
        await cache.set('travel-instructions', {
          content,
          timestamp: Date.now(),
          lastModified: response.headers['last-modified'],
          etag,
          source: 'canada.ca',
          fetchTime: Date.now() - startTime,
        });
      }

      res.setHeader('Cache-Control', 'public, max-age=3600');
      res.setHeader('ETag', etag);
      if (response.headers['last-modified']) {
        res.setHeader('Last-Modified', response.headers['last-modified']);
      }

      return res.json({
        content,
        fresh: true,
        timestamp: new Date().toISOString(),
      });
    } catch (error: unknown) {
      const err = error as Error;
      return res.status(500).json({
        error: 'Failed to fetch travel instructions',
        message: err.message,
      });
    }
  };

  return {
    handleFollowUp,
    handleTravelInstructions,
  };
};
