import { render, screen } from '@testing-library/react';
import React from 'react';
import { describe, expect, it, vi } from 'vitest';

import { TripPlanner, generateTripPlanMessage, TripData, DistanceData } from '../TripPlanner';

vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children: React.ReactNode }) => <div data-testid="sheet">{children}</div>,
  SheetContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  SheetHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
  SheetTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
  SheetTrigger: ({ children }: { children: React.ReactNode }) => (
    <button type="button">{children}</button>
  ),
}));

vi.mock('@/components/ui/select', () => ({
  Select: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectValue: ({ placeholder }: { placeholder?: string }) => <span>{placeholder}</span>,
  SelectItem: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock('@/components/ui/popover', () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PopoverContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock('@/components/ui/calendar', () => ({
  Calendar: () => <div data-testid="calendar" />,
}));

vi.mock('@/components/PlaceAutocompleteSimple', () => ({
  PlaceAutocompleteSimple: ({
    value,
    onChange,
    placeholder,
  }: {
    value: string;
    onChange: (next: string) => void;
    placeholder?: string;
  }) => (
    <input
      data-testid="autocomplete"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
    />
  ),
}));

vi.mock('@/components/ui/checkbox', () => ({
  Checkbox: ({
    checked,
    onCheckedChange,
  }: {
    checked?: boolean;
    onCheckedChange?: (next: boolean) => void;
  }) => (
    <input
      data-testid="checkbox"
      type="checkbox"
      checked={Boolean(checked)}
      onChange={(event) => onCheckedChange?.(event.target.checked)}
    />
  ),
}));

describe('TripPlanner', () => {
  it('renders the planner sheet contents when open', () => {
    render(<TripPlanner onSubmit={vi.fn()} open />);

    expect(screen.getByText('Trip Planner (Beta)')).toBeInTheDocument();
    expect(screen.getByText('Generate Trip Plan')).toBeInTheDocument();
  });

  it('includes cost estimates in the generated trip plan message', () => {
    const tripData: TripData = {
      transportMethod: 'personal-vehicle',
      departureDate: new Date('2024-01-01T00:00:00Z'),
      returnDate: new Date('2024-01-03T00:00:00Z'),
      departureLocation: 'CFB Toronto, Toronto, ON',
      arrivalLocation: 'CFB Ottawa, Ottawa, ON',
      rnqProvided: true,
      travelAuthority: true,
      purpose: 'Training exercise',
      additionalNotes: '',
    };

    const distanceData: DistanceData = {
      distance: { text: '450 km', value: 450000 },
      duration: { text: '4 hours 30 mins', value: 16200 },
      origin: 'Toronto, ON',
      destination: 'Ottawa, ON',
      mode: 'driving',
    };

    const plan = generateTripPlanMessage(tripData, distanceData);

    expect(plan).toContain('💵 **Estimated Costs:**');
    expect(plan).toContain('Incidentals (3 days): $51.90');
    expect(plan).toContain(
      'Use RAG to retrieve the current private-vehicle kilometric rate covering travel between CFB Toronto, Toronto, ON → CFB Ottawa, Ottawa, ON (Ontario). Apply it to 900 km to estimate mileage cost.',
    );
    expect(plan).toContain(
      '**Please combine the RAG-derived kilometric mileage cost with the incidentals above to present the total trip estimate.**',
    );
  });
});
