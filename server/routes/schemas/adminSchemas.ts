import { z } from 'zod';

/**
 * Schema for RAG configuration updates.
 * Security: Validates config structure and prevents injection of invalid values.
 */
export const ragConfigUpdateSchema = z.object({
  config_updates: z
    .record(
      z.string().max(100, 'Config key too long'),
      z.union([
        z.string().max(10000, 'String value too long'),
        z.number(),
        z.boolean(),
        z.null(),
        z.array(z.string().max(1000)),
      ])
    )
    .optional(),
  // Allow top-level config keys as fallback (for backwards compatibility)
}).passthrough().refine(
  (data) => {
    // Security: Prevent prototype pollution
    const dangerousKeys = ['__proto__', 'constructor', 'prototype'];
    const allKeys = Object.keys(data);
    const configKeys = data.config_updates ? Object.keys(data.config_updates) : [];
    return ![...allKeys, ...configKeys].some((key) => dangerousKeys.includes(key));
  },
  { message: 'Invalid configuration key detected' }
);

/**
 * Schema for analytics visits query parameters.
 * Security: Validates date formats and path parameter.
 */
export const analyticsVisitsQuerySchema = z.object({
  startAt: z
    .string()
    .max(50, 'startAt exceeds maximum length')
    .refine(
      (val) => !val || !isNaN(Date.parse(val)),
      'startAt must be a valid date string'
    )
    .optional(),
  endAt: z
    .string()
    .max(50, 'endAt exceeds maximum length')
    .refine(
      (val) => !val || !isNaN(Date.parse(val)),
      'endAt must be a valid date string'
    )
    .optional(),
  path: z
    .string()
    .max(500, 'path exceeds maximum length')
    .optional(),
});
