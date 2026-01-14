/**
 * Utility to extract Class A Reservist section from chat response content.
 */

export interface ClassASectionResult {
  mainContent: string;
  classAContent: string | null;
}

/**
 * Extracts the "For Class A Reservists" section from markdown content.
 * Returns the main content and the Class A section separately for custom rendering.
 */
export function extractClassASection(content: string): ClassASectionResult {
  // Match "**For Class A Reservists:**" (with or without trailing colon) and everything after it
  const regex = /\*\*For Class A Reservists:?\*\*:?\s*([\s\S]*)$/i;
  const match = content.match(regex);

  if (match && match.index !== undefined) {
    const classAContent = match[1].trim();
    const mainContent = content.slice(0, match.index).trim();
    return { mainContent, classAContent };
  }

  return { mainContent: content, classAContent: null };
}
