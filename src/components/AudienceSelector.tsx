import React from 'react';

interface AudienceSelectorProps {
  value: 'general' | 'classA';
  onChange: (value: 'general' | 'classA') => void;
  className?: string;
}

export const AudienceSelector: React.FC<AudienceSelectorProps> = ({
  value,
  onChange,
  className,
}) => {
  return (
    <div className={className}>
      <label className="sr-only" htmlFor="audience-select">
        Audience
      </label>
      <select
        id="audience-select"
        className="h-11 rounded-lg border-2 bg-[var(--card)] text-[var(--text)] border-[var(--border)] px-3 shadow-md hover:shadow-lg transition-all duration-200"
        value={value}
        onChange={(e) => onChange((e.target.value as 'general' | 'classA') || 'general')}
      >
        <option value="general">General</option>
        <option value="classA">Class A</option>
      </select>
    </div>
  );
};

export default AudienceSelector;
