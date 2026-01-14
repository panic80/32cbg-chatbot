import type { Request, Response } from 'express';
import { Client, UnitSystem } from '@googlemaps/google-maps-services-js';
import { DEFAULT_MAPS_TIMEOUT_MS, getEnvNumber } from '../config/constants.js';
import { respondWithError } from '../utils/http.js';

interface MapsControllerConfig {
  googleMapsClient: Client | null;
  config?: { mapsTimeout?: number };
  logger: import('../services/logger.js').Logger;
}

const getTimeout = (config: MapsControllerConfig['config']) => {
  if (config?.mapsTimeout && Number.isFinite(config.mapsTimeout)) {
    return config.mapsTimeout;
  }
  const envTimeout = getEnvNumber('MAPS_TIMEOUT', DEFAULT_MAPS_TIMEOUT_MS);
  return envTimeout || DEFAULT_MAPS_TIMEOUT_MS;
};

export const createMapsController = ({
  googleMapsClient,
  config = {},
  logger,
}: MapsControllerConfig) => {
  const scopedLogger = logger?.child ? logger.child({ scope: 'controller:maps' }) : logger;
  const emit = (level: string, message: string, meta?: unknown) => {
    const loggerFunc = (scopedLogger as unknown as Record<string, unknown>)[level];
    if (typeof loggerFunc === 'function') {
      (loggerFunc as Function)(message, meta);
    }
  };

  const ensureClient = (res: Response): boolean => {
    if (!googleMapsClient) {
      respondWithError(res, {
        status: 503,
        error: 'MapsClientUnavailable',
        message: 'Google Maps service is not configured',
        logger: scopedLogger,
        level: 'warn',
      });
      return false;
    }
    return true;
  };

  const handleDistance = async (req: Request, res: Response) => {
    if (!ensureClient(res)) return;

    const { origin, destination, mode = 'driving' } = req.body;
    const timeoutMs = getTimeout(config);

    emit('info', 'maps.distanceMatrix', { origin, destination, mode, timeoutMs });

    try {
      const response = await googleMapsClient!.distancematrix({
        params: {
          origins: [origin],
          destinations: [destination],
          mode,
          units: UnitSystem.metric,
          key: process.env.GOOGLE_MAPS_API_KEY as string,
        },
        timeout: timeoutMs,
      });

      const data = response.data;
      const element = data?.rows?.[0]?.elements?.[0];

      if (!element) {
        return respondWithError(res, {
          status: 502,
          error: 'MapsResponseMalformed',
          message: 'Distance matrix response missing data',
          logger: scopedLogger,
          level: 'warn',
          details: { origin, destination, mode },
        });
      }

            if (element.status !== 'OK') {
              const errorData = element as unknown as Record<string, unknown>;
              return respondWithError(res, {
                status: 422,
                error: element.status,
                message: (errorData.error_message as string) || 'Failed to calculate distance', 
                logger: scopedLogger,
                level: 'warn',
                details: { origin, destination, mode, elementStatus: element.status },
              });
            }
      interface DistanceResult {
        distance: import('@googlemaps/google-maps-services-js').Distance;
        duration: import('@googlemaps/google-maps-services-js').Duration;
        origin: string;
        destination: string;
        mode: string;
        totalDistance?: number;
        totalDuration?: number;
      }

      const result: DistanceResult = {
        distance: element.distance,
        duration: element.duration,
        origin: data.origin_addresses?.[0] ?? origin,
        destination: data.destination_addresses?.[0] ?? destination,
        mode,
      };

      if (mode === 'driving') {
        const totals = data.rows
          .flatMap((row) => row.elements)
          .reduce(
            (acc, curr) => {
              if (curr.status === 'OK') {
                if (curr.distance?.value) acc.totalDistance += curr.distance.value;
                if (curr.duration?.value) acc.totalDuration += curr.duration.value;
              }
              return acc;
            },
            { totalDistance: 0, totalDuration: 0 },
          );

        result.totalDistance = totals.totalDistance;
        result.totalDuration = totals.totalDuration;
      }

      res.json(result);
    } catch (error: unknown) {
      const err = error as Error & { response?: { status: number } };
      if (err?.response?.status === 403) {
        return respondWithError(res, {
          status: 403,
          error: 'Forbidden',
          message: 'Ensure the Google Maps API key has Distance Matrix API enabled.',
          logger: scopedLogger,
          level: 'warn',
          cause: err,
          details: { origin, destination, mode },
        });
      }

      return respondWithError(res, {
        status: err?.response?.status || 500,
        error: 'FailedToCalculateDistance',
        message: err?.message || 'Failed to calculate distance',
        logger: scopedLogger,
        cause: err,
        details: { origin, destination, mode },
      });
    }
  };

  const handleAutocomplete = async (req: Request, res: Response) => {
    if (!ensureClient(res)) return;

    const { input, sessiontoken, components } = req.query;

    if (!input) {
      return respondWithError(res, {
        status: 400,
        error: 'MissingInput',
        message: 'Input parameter is required',
        logger: scopedLogger,
        level: 'warn',
      });
    }

    emit('info', 'maps.autocomplete', { input, hasSessionToken: Boolean(sessiontoken) });

    try {
      const params: Record<string, unknown> = {
        input: input as string,
        key: process.env.GOOGLE_MAPS_API_KEY as string,
      };

      if (sessiontoken) {
        params.sessiontoken = sessiontoken as string;
      }

      if (components) {
        params.components = components as string[];
      }

      const response = await googleMapsClient!.placeAutocomplete({
        params:
          params as unknown as import('@googlemaps/google-maps-services-js').PlaceAutocompleteRequest['params'],
        timeout: getTimeout(config),
      });

      res.json(response.data);
    } catch (error: unknown) {
      const err = error as Error & { response?: { status: number } };
      if (err?.response?.status === 403) {
        return respondWithError(res, {
          status: 403,
          error: 'Forbidden',
          message: 'Ensure the Google Maps API key has Places API enabled.',
          logger: scopedLogger,
          level: 'warn',
          cause: err,
          details: { input },
        });
      }

      return respondWithError(res, {
        status: err?.response?.status || 500,
        error: 'AutocompleteFailed',
        message: err?.message || 'Failed to fetch autocomplete predictions',
        logger: scopedLogger,
        cause: err,
        details: { input },
      });
    }
  };

  const handlePlaceDetails = async (req: Request, res: Response) => {
    if (!ensureClient(res)) return;

    const { place_id: placeId, sessiontoken } = req.query;

    if (!placeId) {
      return respondWithError(res, {
        status: 400,
        error: 'MissingPlaceId',
        message: 'place_id parameter is required',
        logger: scopedLogger,
        level: 'warn',
      });
    }

    emit('info', 'maps.placeDetails', { placeId });

    try {
      const params: Record<string, unknown> = {
        place_id: placeId as string,
        key: process.env.GOOGLE_MAPS_API_KEY as string,
      };

      if (sessiontoken) {
        params.sessiontoken = sessiontoken as string;
      }

      const response = await googleMapsClient!.placeDetails({
        params:
          params as unknown as import('@googlemaps/google-maps-services-js').PlaceDetailsRequest['params'],
        timeout: getTimeout(config),
      });

      res.json(response.data);
    } catch (error: unknown) {
      const err = error as Error & { response?: { status: number } };
      if (err?.response?.status === 403) {
        return respondWithError(res, {
          status: 403,
          error: 'Forbidden',
          message: 'Ensure the Google Maps API key has Places API enabled.',
          logger: scopedLogger,
          level: 'warn',
          cause: err,
          details: { placeId },
        });
      }

      return respondWithError(res, {
        status: err?.response?.status || 500,
        error: 'PlaceDetailsFailed',
        message: err?.message || 'Failed to fetch place details',
        logger: scopedLogger,
        cause: err,
        details: { placeId },
      });
    }
  };

  return {
    handleDistance,
    handleAutocomplete,
    handlePlaceDetails,
  };
};
