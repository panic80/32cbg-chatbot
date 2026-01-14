import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { DiffPanel } from '@/components/DiffPanel';
import type { DeltaResponse } from '@/types/policy';

describe('DiffPanel', () => {
  it('renders sections when differences exist', () => {
    const delta: DeltaResponse = {
      stricter: [
        {
          policyArea: 'leave',
          dedupeKey: 'leave.sick.waiting-period',
          changeType: 'stricter',
          summary: 'Class A requires 6 months minimum service before benefit X.',
          citations: ['doc1#s1'],
        },
      ],
      looser: [],
      additionalRequirements: [],
      exceptions: [],
      replacements: [],
      notApplicable: [],
      additions: [],
    } as DeltaResponse;
    render(<DiffPanel delta={delta} />);
    expect(screen.getByText(/For Class A Reservists/i)).toBeInTheDocument();
    // Category heading might duplicate badge text; assert summary exists instead
    expect(screen.getAllByText(/stricter/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/requires 6 months minimum/i)).toBeInTheDocument();
  });
});
