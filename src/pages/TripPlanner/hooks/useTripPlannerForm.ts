import { useCallback, useMemo, useState } from 'react';
import { calculateMealEntitlements, calculateTripDurationInDays } from '../utils/calculations';
import type { TripData, MealEntitlementDay } from '../types';

export const defaultTripData: TripData = {
  transportMethod: '',
  departureDate: undefined,
  returnDate: undefined,
  departureLocation: '',
  arrivalLocation: '',
  rnqProvided: true,
  travelAuthority: false,
  purpose: '',
  additionalNotes: '',
};

export const useTripPlannerForm = (initialState: TripData = defaultTripData) => {
  const [tripData, setTripData] = useState<TripData>(() => ({ ...initialState }));

  const updateTripData = useCallback(
    <Key extends keyof TripData>(key: Key, value: TripData[Key]) => {
      setTripData((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const resetTripData = useCallback(() => {
    setTripData({ ...defaultTripData });
  }, []);

  const tripDuration = useMemo(
    () => calculateTripDurationInDays(tripData.departureDate, tripData.returnDate),
    [tripData.departureDate, tripData.returnDate],
  );

  const mealEntitlements: MealEntitlementDay[] = useMemo(
    () => calculateMealEntitlements(tripDuration, tripData.rnqProvided),
    [tripDuration, tripData.rnqProvided],
  );

  const isFormValid = useCallback(
    () =>
      Boolean(
        tripData.transportMethod &&
          tripData.departureDate &&
          tripData.returnDate &&
          tripData.departureLocation &&
          tripData.arrivalLocation &&
          tripData.purpose,
      ),
    [tripData],
  );

  return {
    tripData,
    updateTripData,
    resetTripData,
    tripDuration,
    mealEntitlements,
    isFormValid,
  };
};
