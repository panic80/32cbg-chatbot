import React from 'react';
import { MarkdownRenderer } from '@/components/ui/markdown-renderer';

interface ClassASectionProps {
  content: string;
}

/**
 * Styled container for Class A Reservist-specific information.
 * Renders with an amber border to visually distinguish it from the main response.
 */
export const ClassASection: React.FC<ClassASectionProps> = ({ content }) => {
  if (!content) return null;

  return (
    <div className="mt-4 rounded-md border border-amber-500/30 bg-amber-500/5 p-4">
      <h4 className="text-sm font-semibold text-amber-600 dark:text-amber-400 mb-2 flex items-center gap-2">
        <span className="text-base">⚠</span>
        For Class A Reservists
      </h4>
      <div className="text-sm text-[var(--text)]/90">
        <MarkdownRenderer>{content}</MarkdownRenderer>
      </div>
    </div>
  );
};

export default ClassASection;
