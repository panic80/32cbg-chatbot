import { z } from 'zod';
import { optionalTrimmedString, trimmedString } from './helpers.js';

export const visitEventSchema = z.object({
  path: trimmedString('Path'),
  referrer: optionalTrimmedString('Referrer'),
  sessionId: optionalTrimmedString('Session ID'),
  locale: optionalTrimmedString('Locale'),
  title: optionalTrimmedString('Title'),
  viewport: optionalTrimmedString('Viewport'),
  metadata: z.record(z.any()).optional(),
});
