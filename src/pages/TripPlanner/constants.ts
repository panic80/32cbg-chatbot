import type { TransportMethod } from './types';

export interface TransportMethodOption {
  value: TransportMethod | '';
  label: string;
}

export const TRANSPORT_METHODS: TransportMethodOption[] = [
  { value: 'personal-vehicle', label: 'Personal Vehicle' },
  { value: 'government-vehicle', label: 'Government Vehicle' },
  { value: 'air', label: 'Air Travel' },
  { value: 'train', label: 'Train' },
  { value: 'bus', label: 'Bus' },
  { value: 'rental', label: 'Rental Vehicle' },
  { value: 'other', label: 'Other' },
];

export const TRANSPORT_TO_MODE_MAP: Record<TransportMethod, string> = {
  'personal-vehicle': 'driving',
  'government-vehicle': 'driving',
  air: 'driving',
  train: 'transit',
  bus: 'transit',
  rental: 'driving',
  other: 'driving',
};

export const getTransportLabel = (transportMethod: TransportMethod | '') =>
  TRANSPORT_METHODS.find((method) => method.value === transportMethod)?.label || 'Not specified';
