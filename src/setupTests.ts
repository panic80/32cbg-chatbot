import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

// Basic IndexedDB mock (tests will override with their own detailed mocks)
if (!global.indexedDB) {
  Object.defineProperty(global, 'indexedDB', {
    value: {
      open: vi.fn(),
      deleteDatabase: vi.fn(),
      cmp: vi.fn(),
    },
    writable: true,
  });
}

// Mock fetch only if not already mocked (tests will set their own)
if (!global.fetch) {
  Object.defineProperty(global, 'fetch', {
    value: vi.fn(),
    writable: true,
  });
}

// Mock browser APIs that are not available in the test environment
Object.defineProperty(global, 'ResizeObserver', {
  value: vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  })),
  writable: true,
});

// Mock env variables
Object.defineProperty(window, 'matchMedia', {
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
  writable: true,
});

// Mock EventSource
Object.defineProperty(global, 'EventSource', {
  value: vi.fn().mockImplementation(() => ({
    close: vi.fn(),
    onmessage: null,
    onerror: null,
    onopen: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })),
  writable: true,
});

// Mock the import.meta.env
vi.stubGlobal('import.meta', {
  env: {
    DEV: true,
  },
});
