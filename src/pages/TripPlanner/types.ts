export type TransportMethod =
  | 'personal-vehicle'
  | 'government-vehicle'
  | 'air'
  | 'train'
  | 'bus'
  | 'rental'
  | 'other';

export interface TripData {
  transportMethod: TransportMethod | '';
  departureDate: Date | undefined;
  returnDate: Date | undefined;
  departureLocation: string;
  arrivalLocation: string;
  rnqProvided: boolean;
  travelAuthority: boolean;
  purpose: string;
  additionalNotes: string;
}

export interface DistanceData {
  distance: {
    text: string;
    value: number;
  };
  duration: {
    text: string;
    value: number;
  };
  origin: string;
  destination: string;
  mode: string;
}

export interface MealEntitlementDay {
  dayLabel: string;
  meals: string[];
}

export interface MileageContext {
  locationLabel: string;
  locationLabelLower: string;
  distanceText: string | null;
  roundTripDistanceText: string | null;
  distanceKm: number | null;
  roundTripKm: number | null;
}
