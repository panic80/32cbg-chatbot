/**
 * Format plain text response into markdown
 */
export const formatPlainTextToMarkdown = (text: string): string => {
  let formatted = text;

  // Add main heading if the response starts with an item name
  if (formatted.match(/^[A-Z][a-z]+ is/)) {
    const firstPeriod = formatted.indexOf('.');
    if (firstPeriod > 0) {
      const title = formatted.substring(0, firstPeriod);
      formatted = `## ${title}\n\n${formatted.substring(firstPeriod + 1)}`;
    }
  }

  // Convert numbered sections (e.g., "1. Nature of Enigma") to headings
  formatted = formatted.replace(/(\d+)\.\s*([A-Z][^:]+)(?=[A-Z])/g, '\n\n### $2\n\n');

  // Convert bullet points patterns (e.g., "* Item:" or "• Item:")
  formatted = formatted.replace(/\*\s+([^:]+):/g, '\n• **$1:**');

  // Bold important terms followed by colons
  formatted = formatted.replace(/([A-Z][a-zA-Z\s]+):\s/g, '**$1:** ');

  // Add line breaks between major sections
  formatted = formatted.replace(/\.(\d+\.)/g, '.\n\n$1');

  // Ensure proper spacing after periods
  formatted = formatted.replace(/\.([A-Z])/g, '.\n\n$1');

  // Clean up excessive line breaks
  formatted = formatted.replace(/\n{3,}/g, '\n\n');

  return formatted.trim();
};
