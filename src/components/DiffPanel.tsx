import React from 'react';
import type { DeltaResponse, DeltaItem } from '@/types/policy';

interface DiffPanelProps {
  delta: DeltaResponse;
}

const Section: React.FC<{ title: string; items: DeltaItem[]; badgeClass: string }> = ({
  title,
  items,
  badgeClass,
}) => {
  if (!items || items.length === 0) return null;
  return (
    <div className="mt-4 border-t border-[var(--border)] pt-3">
      <h4 className="text-sm font-semibold text-[var(--text)] mb-2">{title}</h4>
      <ul className="list-disc ml-5 space-y-1">
        {items.map((it, idx) => (
          <li key={`${it.policyArea}:${it.dedupeKey}:${idx}`} className="text-[var(--text)]/90">
            <span className={`inline-block text-[10px] px-1.5 py-0.5 rounded ${badgeClass} mr-2`}>
              {it.changeType}
            </span>
            {it.summary}
          </li>
        ))}
      </ul>
    </div>
  );
};

export const DiffPanel: React.FC<DiffPanelProps> = ({ delta }) => {
  const total =
    (delta.stricter?.length || 0) +
    (delta.looser?.length || 0) +
    (delta.additionalRequirements?.length || 0) +
    (delta.exceptions?.length || 0) +
    (delta.replacements?.length || 0) +
    (delta.notApplicable?.length || 0) +
    (delta.additions?.length || 0);

  if (!total) return null;

  return (
    <div className="mt-4 rounded-md border border-[var(--border)] bg-[var(--card)] p-3">
      <h3 className="text-sm font-semibold text-[var(--text)]">For Class A Reservists</h3>
      <Section
        title="Stricter"
        items={delta.stricter || []}
        badgeClass="bg-red-500/10 text-red-500 border border-red-500/20"
      />
      <Section
        title="Looser"
        items={delta.looser || []}
        badgeClass="bg-green-500/10 text-green-500 border border-green-500/20"
      />
      <Section
        title="Additional Requirements"
        items={delta.additionalRequirements || []}
        badgeClass="bg-amber-500/10 text-amber-600 border border-amber-500/20"
      />
      <Section
        title="Exceptions"
        items={delta.exceptions || []}
        badgeClass="bg-indigo-500/10 text-indigo-500 border border-indigo-500/20"
      />
      <Section
        title="Replacements"
        items={delta.replacements || []}
        badgeClass="bg-slate-500/10 text-slate-500 border border-slate-500/20"
      />
      <Section
        title="Not Applicable"
        items={delta.notApplicable || []}
        badgeClass="bg-gray-500/10 text-gray-500 border border-gray-500/20"
      />
      <Section
        title="Additions"
        items={delta.additions || []}
        badgeClass="bg-blue-500/10 text-blue-500 border border-blue-500/20"
      />
    </div>
  );
};

export default DiffPanel;
