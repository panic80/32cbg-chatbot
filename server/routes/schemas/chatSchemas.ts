import { z } from 'zod';
import { optionalTrimmedString, trimmedString } from './helpers.js';

export const geminiGenerationSchema = z.object({
  prompt: trimmedString('Prompt'),
  model: optionalTrimmedString('Model'),
});

export const standardChatSchema = z.object({
  message: trimmedString('Message'),
  model: trimmedString('Model'),
  provider: trimmedString('Provider'),
});

export const ragChatSchema = z.object({
  message: trimmedString('Message'),
  model: optionalTrimmedString('Model'),
  provider: optionalTrimmedString('Provider'),
  chatHistory: z.array(z.any()).optional(),
  conversationId: optionalTrimmedString('Conversation ID'),
  useRAG: z.boolean().optional(),
  audience: optionalTrimmedString('Audience'),
});

export const streamingChatSchema = ragChatSchema.extend({
  shortAnswerMode: z.boolean().optional(),
  useHybridSearch: z.boolean().optional(),
  reasoningEffort: optionalTrimmedString('Reasoning effort'),
  responseVerbosity: optionalTrimmedString('Response verbosity'),
  modelMode: z.enum(['fast', 'smart']).optional(),
});
