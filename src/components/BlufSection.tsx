import React from 'react';

interface BlufSectionProps {
  content: string;
}

/**
 * Styled container for BLUF (Bottom Line Up Front) information.
 * Renders with a blue border to visually distinguish it at the top of responses.
 */
export const BlufSection: React.FC<BlufSectionProps> = ({ content }) => {
  if (!content) return null;

  return (
    <div className="mb-4 rounded-md border border-blue-500/30 bg-blue-500/5 p-4">
      <h4 className="text-sm font-semibold text-blue-600 dark:text-blue-400 mb-2 flex items-center gap-2">
        <span className="text-base">📌</span>
        Bottom Line
      </h4>
      <div className="text-[var(--text)]">{content}</div>
    </div>
  );
};

export default BlufSection;
