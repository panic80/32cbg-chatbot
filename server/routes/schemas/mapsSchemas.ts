import { z } from 'zod';
import { optionalTrimmedString, trimmedString } from './helpers.js';

export const distanceRequestSchema = z.object({
  origin: trimmedString('Origin'),
  destination: trimmedString('Destination'),
  mode: optionalTrimmedString('Mode').refine(
    (value) => !value || ['driving', 'walking', 'bicycling', 'transit'].includes(value),
    'Mode must be one of driving, walking, bicycling, or transit.',
  ),
});

/**
 * Schema for autocomplete query parameters.
 * Security: Validates input length and format to prevent abuse.
 */
export const autocompleteQuerySchema = z.object({
  input: z
    .string()
    .min(1, 'Input parameter is required')
    .max(500, 'Input exceeds maximum length')
    .transform((v) => v.trim()),
  sessiontoken: z
    .string()
    .max(100, 'Session token exceeds maximum length')
    .optional(),
  components: z
    .string()
    .max(200, 'Components string exceeds maximum length')
    .optional(),
});

/**
 * Schema for place details query parameters.
 * Security: Validates place_id format to prevent injection.
 */
export const placeDetailsQuerySchema = z.object({
  place_id: z
    .string()
    .min(1, 'place_id parameter is required')
    .max(300, 'place_id exceeds maximum length')
    .regex(/^[a-zA-Z0-9_-]+$/, 'Invalid place_id format'),
  sessiontoken: z
    .string()
    .max(100, 'Session token exceeds maximum length')
    .optional(),
});
