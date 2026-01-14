/**
 * Source formatting utilities
 * Extracted from useStreamingChat hook for better organization and testability
 */

import type { Source } from '@/types';

interface RawSource {
  id?: string;
  reference?: string;
  title?: string;
  url?: string;
  content?: string;
  text?: string;
  section?: string;
  page?: number;
  score?: number;
  source?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Convert raw source objects from API to Source type
 */
export function toSources(eventSources: RawSource[] = []): Source[] {
  return eventSources.map((source, index) => ({
    id: source.id || source.reference || source.title || source.url || `stream-source-${index}`,
    text: source.content || source.text || '',
    title: source.title,
    url: source.url,
    section: source.section,
    page: source.page,
    score: source.score,
    reference: source.source || source.reference || source.title || '',
    metadata: source.metadata,
  }));
}

/**
 * Check if a string looks like a file path or URL
 */
function looksLikePath(value: string): boolean {
  return /[\\/]/.test(value) || /^[a-z]+:\/\//i.test(value);
}

/**
 * Convert string to title case
 */
function toTitleCase(value: string): string {
  return value.replace(/\b([a-zA-Z])/g, (match) => match.toUpperCase());
}

/**
 * Sanitize a filename to extract a readable label
 */
function sanitizeFilename(value: string): string {
  const withoutPath = value.split(/[\\\/]/).pop() || value;
  const withoutExt = withoutPath.replace(/\.[a-z0-9]+$/i, '');
  const normalized = withoutExt
    .replace(/[_\-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return normalized ? toTitleCase(normalized) : withoutExt;
}

interface CandidateConfig {
  value: string | undefined;
  sanitize: boolean;
}

/**
 * Derive a human-readable label for a source
 * Searches through title/metadata candidates, falling back to sanitized filenames
 */
export function deriveSourceLabel(source: Source, index: number): string {
  const candidates: CandidateConfig[] = [
    // Primary candidates - use directly if not a path
    { value: source.title, sanitize: false },
    { value: source.metadata?.title, sanitize: false },
    { value: source.metadata?.documentTitle, sanitize: false },
    { value: source.metadata?.displayTitle, sanitize: false },
    { value: source.metadata?.display_name, sanitize: false },
    { value: source.metadata?.displayName, sanitize: false },
    { value: source.metadata?.catalogTitle, sanitize: false },
    { value: source.metadata?.catalog_title, sanitize: false },
    { value: source.metadata?.canonicalTitle, sanitize: false },
    { value: source.metadata?.canonical_title, sanitize: false },
    { value: source.metadata?.document_name, sanitize: false },
    { value: source.metadata?.documentName, sanitize: false },
    { value: source.metadata?.sourceTitle, sanitize: false },
    { value: source.metadata?.source_name, sanitize: false },
    { value: source.metadata?.sourceName, sanitize: false },
    { value: source.metadata?.name, sanitize: false },
    { value: source.reference, sanitize: false },
    { value: source.section, sanitize: false },
    // Fallback candidates - require sanitization
    { value: source.metadata?.original_filename, sanitize: true },
    { value: source.metadata?.original_name, sanitize: true },
    { value: source.metadata?.filename, sanitize: true },
    { value: source.metadata?.file_name, sanitize: true },
    { value: source.metadata?.source, sanitize: true },
    { value: source.reference, sanitize: true },
    { value: source.url, sanitize: true },
  ];

  for (const { value, sanitize } of candidates) {
    if (typeof value !== 'string') continue;
    const trimmed = value.trim();
    if (!trimmed) continue;

    if (sanitize) {
      const cleaned = sanitizeFilename(trimmed);
      if (cleaned) return cleaned;
    } else {
      if (!looksLikePath(trimmed)) return trimmed;
    }
  }

  return `Source ${index + 1}`;
}
