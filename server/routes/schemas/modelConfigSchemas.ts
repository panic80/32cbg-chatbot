import { z } from 'zod';

/**
 * Valid AI providers for model configuration
 */
const providerSchema = z.enum(['openai', 'google', 'anthropic', 'openrouter']);

/**
 * Valid model designations for operation models
 */
const designationSchema = z.enum(['fast', 'smart']);

/**
 * Schema for a model reference (provider + model name)
 */
const modelRefSchema = z.object({
  provider: providerSchema,
  model: z.string().min(1, 'Model name is required').max(100, 'Model name too long'),
});

/**
 * Schema for operation model configuration
 */
const operationModelsSchema = z.object({
  responseGeneration: designationSchema.optional(),
  hydeExpansion: designationSchema.optional(),
  queryRewriting: designationSchema.optional(),
  followUpGeneration: designationSchema.optional(),
}).strict();

/**
 * Schema for POST /api/admin/model-config request body
 */
export const modelConfigUpdateSchema = z.object({
  fastModel: modelRefSchema,
  smartModel: modelRefSchema,
  operationModels: operationModelsSchema.optional(),
}).strict();

export type ModelConfigUpdateRequest = z.infer<typeof modelConfigUpdateSchema>;
