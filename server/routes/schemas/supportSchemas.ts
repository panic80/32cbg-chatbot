import { z } from 'zod';
import { optionalTrimmedString, trimmedString } from './helpers.js';

export const followUpRequestSchema = z.object({
  userQuestion: trimmedString('User question'),
  aiResponse: trimmedString('AI response'),
  model: optionalTrimmedString('Model').default('gemini-2.0-flash'),
  provider: optionalTrimmedString('Provider').default('google'),
});
