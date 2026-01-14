/**
 * Utility to extract BLUF (Bottom Line Up Front) section from chat response content.
 */

export interface BlufSectionResult {
  blufContent: string | null;
  remainingContent: string;
}

/**
 * Extracts the "BOTTOM LINE:" section from the beginning of markdown content.
 * Returns the BLUF content and remaining content separately for custom rendering.
 */
export function extractBlufSection(content: string): BlufSectionResult {
  // Match "**BOTTOM LINE:**" at the start and content until the first blank line
  const regex = /^\*\*BOTTOM LINE:\*\*\s*(.+?)\n\n/is;
  const match = content.match(regex);

  if (match) {
    return {
      blufContent: match[1].trim(),
      remainingContent: content.slice(match[0].length).trim(),
    };
  }

  return { blufContent: null, remainingContent: content };
}
