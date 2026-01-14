import { z } from 'zod';

export const trimmedString = (field: string, min: number = 1) =>
  z
    .string({
      required_error: `${field} is required.`,
      invalid_type_error: `${field} must be a string.`,
    })
    .transform((value) => value.trim())
    .refine((value) => value.length >= min, `${field} must be a non-empty string.`);

export const optionalTrimmedString = (field: string) =>
  z
    .string({
      invalid_type_error: `${field} must be a string.`,
    })
    .transform((value) => value.trim())
    .optional();
