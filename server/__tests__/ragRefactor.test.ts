import { createRagService } from '../services/RagService.js';
import { createIngestionController } from '../controllers/ingestionController.js';
import { describe, it, expect, vi } from 'vitest';

describe('RagService Refactor', () => {
  it('should create RagService instance', () => {
    const service = createRagService({
      config: {} as unknown as Parameters<typeof createRagService>[0]['config'],
      logger: console as unknown as Parameters<typeof createRagService>[0]['logger'],
    });
    expect(service).toHaveProperty('ingest');
    expect(service).toHaveProperty('ingestCanadaCa');
    expect(service).toHaveProperty('getProgressStream');
  });

  it('should create IngestionController instance with injected service', () => {
    const mockService = {
      ingest: vi.fn(),
      ingestCanadaCa: vi.fn(),
      getProgressStream: vi.fn(),
    } as unknown as Parameters<typeof createIngestionController>[0]['ragService'];
    const controller = createIngestionController({
      ragService: mockService,
      logger: console as unknown as Parameters<typeof createIngestionController>[0]['logger'],
    });
    expect(controller).toHaveProperty('handleIngest');
    expect(controller).toHaveProperty('handleCanadaCaIngest');
    expect(controller).toHaveProperty('handleProgress');
  });
});
