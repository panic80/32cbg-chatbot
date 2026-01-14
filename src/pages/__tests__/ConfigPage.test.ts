import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

declare global {
  // eslint-disable-next-line no-var
  var $RefreshReg$: () => void;
  // eslint-disable-next-line no-var
  var $RefreshSig$: () => () => void;
  // eslint-disable-next-line no-var
  var __vite_plugin_react_preamble_installed__: boolean;
}

globalThis.$RefreshReg$ = () => {};
globalThis.$RefreshSig$ = () => () => {};
globalThis.__vite_plugin_react_preamble_installed__ = true;

let ConfigPage: typeof import('@/pages/ConfigPage').default;
let fetchMock: import('vitest').Mock;

vi.mock('react-router-dom', () => {
  const React = require('react');
  const navigate = vi.fn();
  return {
    MemoryRouter: ({ children }: { children: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    useNavigate: () => navigate,
    Link: ({ children, to }: { children: React.ReactNode; to: string }) =>
      React.createElement('a', { href: to }, children),
  };
});

vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}));

vi.mock('@/components/IngestionConsole', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: ({ url }: { url: string }) =>
      url ? React.createElement('div', { 'data-testid': 'ingestion-console' }, url) : null,
  };
});

vi.mock('@/components/ui/tabs', () => {
  type TabsState = {
    active: string;
    setActive: (value: string) => void;
  };

  const TabsContext = React.createContext<TabsState | null>(null);

  const Tabs = ({
    defaultValue,
    value,
    onValueChange,
    children,
  }: {
    defaultValue?: string;
    value?: string;
    onValueChange?: (v: string) => void;
    children: React.ReactNode;
  }) => {
    const [internalValue, setInternalValue] = React.useState<string>(value ?? defaultValue ?? '');
    const active = value ?? internalValue;
    const setActive = (next: string) => {
      setInternalValue(next);
      onValueChange?.(next);
    };

    return React.createElement(TabsContext.Provider, { value: { active, setActive } }, children);
  };

  const TabsList = ({ children, ...rest }: { children: React.ReactNode; [key: string]: unknown }) =>
    React.createElement('div', { role: 'tablist', ...rest }, children);

  const TabsTrigger = ({
    value,
    children,
    ...rest
  }: {
    value: string;
    children: React.ReactNode;
    [key: string]: unknown;
  }) => {
    const ctx = React.useContext(TabsContext);
    if (!ctx) {
      throw new Error('TabsTrigger must be used within Tabs');
    }
    const { active, setActive } = ctx;
    return React.createElement(
      'button',
      {
        type: 'button',
        role: 'tab',
        'aria-selected': active === value,
        onClick: () => setActive(value),
        ...rest,
      },
      children,
    );
  };

  const TabsContent = ({
    value,
    children,
    ...rest
  }: {
    value: string;
    children: React.ReactNode;
    [key: string]: unknown;
  }) => {
    const ctx = React.useContext(TabsContext);
    if (!ctx) {
      throw new Error('TabsContent must be used within Tabs');
    }
    if (ctx.active !== value) {
      return null;
    }
    return React.createElement('div', rest, children);
  };

  return { Tabs, TabsList, TabsTrigger, TabsContent };
});

describe('ConfigPage', () => {
  beforeEach(async () => {
    localStorage.clear();
    vi.clearAllMocks();
    fetchMock = vi.fn();
    Object.defineProperty(globalThis, 'fetch', { value: fetchMock, writable: true });
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    ConfigPage = (await import('@/pages/ConfigPage')).default;
  });

  it('shows the default active model information', async () => {
    render(React.createElement(MemoryRouter, null, React.createElement(ConfigPage)));

    const activeModelLabel = await screen.findByText(/Active Model:/);
    const activeModelRow = activeModelLabel.parentElement;
    expect(activeModelRow).not.toBeNull();
    expect(activeModelRow).toHaveTextContent('GPT-5 Mini');
    await waitFor(() => {
      expect(screen.getByText(/Model ID:/)).toHaveTextContent('gpt-5-mini');
    });
  });

  it('surfaces unsaved changes when selecting a different model and persists on save', async () => {
    render(React.createElement(MemoryRouter, null, React.createElement(ConfigPage)));

    fireEvent.click(screen.getByRole('tab', { name: 'Google' }));

    const alternativeModel = await screen.findByText('Gemini 2.5 Pro');
    fireEvent.click(alternativeModel);

    expect(await screen.findByText(/You have unsaved changes/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(localStorage.getItem('selectedLLMModel')).toBe('gemini-2.5-pro');
      expect(localStorage.getItem('selectedLLMProvider')).toBe('google');
    });

    await waitFor(() => {
      expect(screen.queryByText(/You have unsaved changes/)).not.toBeInTheDocument();
    });
  });

  it('loads database stats and renders indexed sources', async () => {
    const statsResponse = {
      total_documents: 25,
      total_chunks: 80,
      total_sources: 2,
      last_ingested_at: '2025-01-01T10:00:00Z',
    };

    const sourcesResponse = {
      items: [
        {
          source_id: 'source-1',
          title: 'Handbook Overview',
          canonical_url: 'https://example.com/handbook/',
          chunk_count: 30,
          document_count: 10,
          last_ingested_at: '2025-01-02T12:30:00Z',
        },
        {
          source_id: 'source-2',
          title: 'FAQ',
          canonical_url: null,
          chunk_count: 5,
          document_count: 5,
          last_ingested_at: null,
        },
      ],
      total: 2,
      page: 1,
      page_size: 100,
    };

    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : (input as Request).url;

      if (url === '/api/v2/sources/stats') {
        return Promise.resolve({
          ok: true,
          json: async () => statsResponse,
        });
      }

      if (url.startsWith('/api/v2/sources?page=1&page_size=100')) {
        return Promise.resolve({
          ok: true,
          json: async () => sourcesResponse,
        });
      }

      return Promise.resolve({
        ok: true,
        json: async () => ({}),
      });
    });

    render(React.createElement(MemoryRouter, null, React.createElement(ConfigPage)));

    fireEvent.click(screen.getByRole('tab', { name: /database/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/v2/sources/stats', expect.anything());
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/v2/sources?page=1&page_size=100',
        expect.anything(),
      );
    });

    expect(await screen.findByText('Handbook Overview')).toBeInTheDocument();
    expect(screen.getByText('10 docs')).toBeInTheDocument();
    expect(screen.getByText('30 chunks')).toBeInTheDocument();
    expect(screen.getByText('example.com/handbook')).toBeInTheDocument();

    const totalSourcesLabel = screen.getByText('Total Sources');
    expect(totalSourcesLabel.parentElement).toHaveTextContent('2');

    // Second source lacks a last ingested timestamp and should fall back to "Unknown"
    expect(screen.getAllByText('Unknown').length).toBeGreaterThanOrEqual(1);
  });

  it('clears ingestion progress when the ingest request fails', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : (input as Request).url;

      if (url === '/api/v2/ingest') {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: async () => ({ message: 'Unable to ingest' }),
        });
      }

      if (url === '/api/rag/ingest') {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: async () => ({ message: 'Unable to ingest' }),
        });
      }

      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({}),
      });
    });

    render(React.createElement(MemoryRouter, null, React.createElement(ConfigPage)));

    fireEvent.click(screen.getByRole('tab', { name: 'URL Ingestion' }));

    const urlField = screen.getByLabelText('Enter URL to Ingest');
    fireEvent.change(urlField, { target: { value: 'https://example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Ingest URL' }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/v2/ingest', expect.any(Object));
    });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/rag/ingest', expect.any(Object));
    });

    await waitFor(() => {
      expect(screen.queryByTestId('ingestion-console')).not.toBeInTheDocument();
    });
  });

  it('fetches chat logs when the Logs tab is opened', async () => {
    const logsResponse = {
      data: [
        {
          id: 1,
          askedAt: '2024-03-01T12:00:00.000Z',
          question: 'How do I submit my travel claim?',
          answer: 'You can submit claims through the secure travel portal.',
          conversationId: 'conv-123',
          model: 'gpt-5-mini',
          provider: 'openai',
          ragEnabled: true,
          shortAnswerMode: false,
          metadata: { route: '/api/v2/chat' },
        },
      ],
      pagination: {
        limit: 20,
        offset: 0,
        hasMore: true,
        nextOffset: 20,
      },
    };

    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : (input as Request).url;

      if (url.startsWith('/api/admin/chat-logs?')) {
        return Promise.resolve({
          ok: true,
          json: async () => logsResponse,
        });
      }

      return Promise.resolve({
        ok: true,
        json: async () => ({}),
      });
    });

    render(React.createElement(MemoryRouter, null, React.createElement(ConfigPage)));

    fireEvent.click(screen.getByRole('tab', { name: /logs/i }));

    await waitFor(() => {
      const calls = fetchMock.mock.calls as Array<[RequestInfo | URL]>;
      expect(
        calls.some(([requestUrl]) => {
          const urlString =
            typeof requestUrl === 'string'
              ? requestUrl
              : requestUrl instanceof URL
                ? requestUrl.toString()
                : ((requestUrl as Request).url ?? requestUrl.toString());
          return (
            typeof urlString === 'string' &&
            urlString.startsWith('/api/admin/chat-logs?limit=20&offset=0')
          );
        }),
      ).toBe(true);
    });

    expect(await screen.findByText('How do I submit my travel claim?')).toBeInTheDocument();
    expect(screen.getByText('RAG: Yes')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
  });

  it('fetches visit analytics summary when the Logs tab opens', async () => {
    const visitAnalyticsResponse = {
      data: {
        totalVisits: 42,
        firstVisit: '2025-01-01T12:00:00Z',
        lastVisit: '2025-01-07T09:30:00Z',
        dailyCounts: [
          { date: '2025-01-01', count: 5 },
          { date: '2025-01-02', count: 7 },
        ],
      },
      filters: {
        startAt: null,
        endAt: null,
        path: null,
      },
    };

    const logsResponse = {
      data: [],
      pagination: {
        limit: 20,
        offset: 0,
        hasMore: false,
        nextOffset: null,
      },
    };

    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url =
        typeof input === 'string'
          ? input
          : input instanceof URL
            ? input.toString()
            : (input as Request).url;

      if (url.startsWith('/api/admin/analytics/visits')) {
        return Promise.resolve({
          ok: true,
          json: async () => visitAnalyticsResponse,
        });
      }

      if (url.startsWith('/api/admin/chat-logs?')) {
        return Promise.resolve({
          ok: true,
          json: async () => logsResponse,
        });
      }

      return Promise.resolve({
        ok: true,
        json: async () => ({}),
      });
    });

    render(React.createElement(MemoryRouter, null, React.createElement(ConfigPage)));

    fireEvent.click(screen.getByRole('tab', { name: /logs/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/admin/analytics/visits', expect.anything());
    });

    await screen.findByText('Visit Analytics');
    const totalVisitsRow = screen.getByText(/Total visits/i).parentElement;
    expect(totalVisitsRow).toHaveTextContent('42');
    expect(screen.getByText(/Last 7 days/i)).toBeInTheDocument();
  });
});
