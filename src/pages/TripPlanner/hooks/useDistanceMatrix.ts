import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient, ApiError } from '@/api/client';
import { TRIP_DISTANCE_DEBOUNCE_MS } from '@/constants/travel';
import { TRANSPORT_TO_MODE_MAP } from '../constants';
import type { DistanceData, TripData, TransportMethod } from '../types';

const resolveMode = (method: TransportMethod | '') =>
  (method && TRANSPORT_TO_MODE_MAP[method]) || 'driving';

export const useDistanceMatrix = (tripData: TripData) => {
  const [distanceData, setDistanceData] = useState<DistanceData | null>(null);
  const [distanceError, setDistanceError] = useState<string | null>(null);
  const [isLoadingDistance, setIsLoadingDistance] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchDistance = useCallback(
    async (overrideTransportMethod?: TransportMethod | '') => {
      const { departureLocation, arrivalLocation, transportMethod } = tripData;
      if (!departureLocation || !arrivalLocation) {
        setDistanceData(null);
        return;
      }

      setIsLoadingDistance(true);
      setDistanceError(null);

      try {
        const mode = resolveMode(overrideTransportMethod ?? transportMethod);

        const data = await apiClient.postJson<DistanceData>(
          '/api/maps/distance',
          {
            origin: departureLocation,
            destination: arrivalLocation,
            mode,
          },
          {
            headers: {
              'Content-Type': 'application/json',
            },
          },
        );

        setDistanceData(data);
      } catch (error: unknown) {
        console.error('Error fetching distance:', error);
        if (error instanceof ApiError) {
          const errorData = error.data as Record<string, unknown> | undefined;
          const message =
            typeof errorData?.error === 'string'
              ? errorData.error
              : error.statusText || error.message;
          setDistanceError(message || 'Failed to calculate distance');
        } else if (error instanceof Error) {
          setDistanceError(error.message);
        } else {
          setDistanceError('Failed to calculate distance');
        }
        setDistanceData(null);
      } finally {
        setIsLoadingDistance(false);
      }
    },
    [tripData],
  );

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    const { departureLocation, arrivalLocation } = tripData;
    if (!departureLocation || !arrivalLocation) {
      setDistanceData(null);
      return;
    }

    debounceRef.current = setTimeout(() => {
      fetchDistance();
    }, TRIP_DISTANCE_DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [
    tripData,
    tripData.departureLocation,
    tripData.arrivalLocation,
    tripData.transportMethod,
    fetchDistance,
  ]);

  return {
    distanceData,
    distanceError,
    isLoadingDistance,
    refreshDistance: fetchDistance,
  };
};
