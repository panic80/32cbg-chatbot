/**
 * Input validation and sanitization utilities
 * Consolidates duplicate validation logic from routes and API clients
 */

/**
 * Parse a value as a number with optional constraints
 * @param value - Value to parse (string, number, or unknown)
 * @param options - Parsing options (fallback, min, max)
 * @returns Parsed number within constraints, or fallback if invalid
 */
export const parseNumber = (
  value: unknown,
  {
    fallback,
    min = 0,
    max = Number.MAX_SAFE_INTEGER,
  }: { fallback: number; min?: number; max?: number },
): number => {
  const parsed = typeof value === 'string' ? Number.parseInt(value, 10) : Number(value);
  if (Number.isNaN(parsed) || !Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(Math.max(parsed, min), max);
};

/**
 * Parse a value as a floating-point number
 * More permissive than parseNumber - accepts decimals
 * @param value - Value to parse
 * @param fallback - Default value if parsing fails
 * @returns Parsed number or fallback
 */
export const toNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseFloat(value);
    if (!Number.isNaN(parsed) && Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
};

/**
 * Sanitize a string value - trim whitespace and return undefined if empty
 * @param value - Value to sanitize
 * @returns Trimmed string or undefined if invalid/empty
 */
export const sanitizeString = (value: unknown): string | undefined => {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

/**
 * Convert a value to string or undefined
 * Alias for sanitizeString with more explicit naming
 * @param value - Value to convert
 * @returns Trimmed string or undefined
 */
export const toStringOrUndefined = (value: unknown): string | undefined => {
  return sanitizeString(value);
};

/**
 * Parse a value as a boolean
 * Accepts: true/false, 'true'/'false', 1/0, '1'/'0'
 * @param value - Value to parse
 * @param fallback - Default value if parsing fails
 * @returns Parsed boolean or fallback
 */
export const parseBoolean = (value: unknown, fallback = false): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    const lower = value.toLowerCase().trim();
    if (lower === 'true' || lower === '1') return true;
    if (lower === 'false' || lower === '0') return false;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  return fallback;
};
