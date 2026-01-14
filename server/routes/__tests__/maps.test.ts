import express from 'express';
import request from 'supertest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import createMapsRoutes from '../maps.js';
import { DEFAULT_MAPS_TIMEOUT_MS } from '../../config/constants.js';

describe('maps routes', () => {
  const originalEnv = process.env.MAPS_TIMEOUT;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    process.env.MAPS_TIMEOUT = originalEnv;
  });

  const buildApp = (overrides: Partial<{ timeout: number }> = {}) => {
    const timeout = overrides.timeout ?? DEFAULT_MAPS_TIMEOUT_MS;
    process.env.MAPS_TIMEOUT = String(timeout);

    const distancematrix = vi.fn().mockResolvedValue({
      data: {
        rows: [
          {
            elements: [
              {
                status: 'OK',
                distance: { text: '1 km', value: 1000 },
                duration: { text: '1 min', value: 60 },
              },
            ],
          },
        ],
        origin_addresses: ['Origin'],
        destination_addresses: ['Destination'],
      },
    });

    const placeAutocomplete = vi
      .fn()
      .mockResolvedValue({ data: { predictions: [], status: 'OK' } });
    const placeDetails = vi
      .fn()
      .mockResolvedValue({ data: { result: { formatted_address: 'Origin' } } });

    const app = express();
    app.use(express.json());
    app.use(
      createMapsRoutes({
        rateLimiter: (_req, _res, next) => next(),
        googleMapsClient: { distancematrix, placeAutocomplete, placeDetails },
      }),
    );

    return { app, distancematrix };
  };

  it('uses the configured timeout for distance calculations', async () => {
    const customTimeout = 12_345;
    const { app, distancematrix } = buildApp({ timeout: customTimeout });

    await request(app)
      .post('/api/maps/distance')
      .send({ origin: 'Origin', destination: 'Destination' })
      .expect(200);

    expect(distancematrix).toHaveBeenCalledTimes(1);
    expect(distancematrix).toHaveBeenCalledWith(
      expect.objectContaining({ timeout: customTimeout }),
    );
  });

  it('falls back to the default timeout when none is provided', async () => {
    process.env.MAPS_TIMEOUT = '';
    const { app, distancematrix } = buildApp({ timeout: DEFAULT_MAPS_TIMEOUT_MS });

    await request(app)
      .post('/api/maps/distance')
      .send({ origin: 'Origin', destination: 'Destination' })
      .expect(200);

    expect(distancematrix).toHaveBeenCalledWith(
      expect.objectContaining({ timeout: DEFAULT_MAPS_TIMEOUT_MS }),
    );
  });

  it('returns 400 when origin or destination missing', async () => {
    const { app } = buildApp();

    const response = await request(app)
      .post('/api/maps/distance')
      .send({ origin: '', destination: '' });

    expect(response.status).toBe(400);
    expect(response.body.message).toBe('Validation failed');
    expect(response.body.details?.[0]?.message).toMatch(/Origin must be a non-empty string/i);
  });
});
