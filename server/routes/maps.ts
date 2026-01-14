import { Router } from 'express';
import { validateRequest, validateQuery } from '../middleware/validate.js';
import {
  distanceRequestSchema,
  autocompleteQuerySchema,
  placeDetailsQuerySchema,
} from './schemas/mapsSchemas.js';
import { getLogger } from '../services/logger.js';
import { createMapsController } from '../controllers/mapsController.js';
import { Client } from '@googlemaps/google-maps-services-js';

interface MapsRoutesConfig {
  rateLimiter: import('express').RequestHandler;
  googleMapsClient: Client | null;
  config?: { mapsTimeout?: number };
}

const createMapsRoutes = ({ rateLimiter, googleMapsClient, config = {} }: MapsRoutesConfig) => {
  const router = Router();
  const logger = getLogger('routes:maps');
  const controller = createMapsController({ googleMapsClient, config, logger });

  const validateDistance = validateRequest(distanceRequestSchema);
  const validateAutocomplete = validateQuery(autocompleteQuerySchema);
  const validatePlaceDetails = validateQuery(placeDetailsQuerySchema);

  router.post('/api/maps/distance', rateLimiter, validateDistance, controller.handleDistance);

  // Security: Added input validation for autocomplete and place details endpoints
  router.get('/api/maps/autocomplete', rateLimiter, validateAutocomplete, controller.handleAutocomplete);
  router.get('/api/maps/place-details', rateLimiter, validatePlaceDetails, controller.handlePlaceDetails);

  return router;
};

export default createMapsRoutes;
