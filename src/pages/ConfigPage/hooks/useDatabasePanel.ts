import { useState, useMemo, useCallback, useRef } from 'react';
import { apiClient, ApiError } from '@/api/client';
import type { DatabaseSource, DatabaseStats } from '../types';

type SourceSort = 'date' | 'count' | 'name';

const MAX_CAPACITY = 100_000;

const normaliseString = (value: string | null | undefined) => (value ?? '').toLowerCase();

const compareBy = (sortBy: SourceSort) => {
  switch (sortBy) {
    case 'name':
      return (a: DatabaseSource, b: DatabaseSource) => a.label.localeCompare(b.label);
    case 'count':
      return (a: DatabaseSource, b: DatabaseSource) => (b.chunkCount || 0) - (a.chunkCount || 0);
    case 'date':
    default:
      return (a: DatabaseSource, b: DatabaseSource) => {
        const dateA = a.lastIngestedAt ? new Date(a.lastIngestedAt).getTime() : 0;
        const dateB = b.lastIngestedAt ? new Date(b.lastIngestedAt).getTime() : 0;
        return (Number.isFinite(dateB) ? dateB : 0) - (Number.isFinite(dateA) ? dateA : 0);
      };
  }
};

const normaliseSources = (list: unknown[]): DatabaseSource[] =>
  list
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') // item is still somewhat any-like for dynamic access
    .map((item) => ({
      id:
        (item.id as string) ??
        (item.source_id as string) ??
        `${normaliseString(((item.label ?? item.title ?? item.name) as string) ?? '')}-${normaliseString(
          ((item.canonicalUrl ?? item.canonical_url ?? item.url) as string) ?? '',
        )}`,
      label: ((item.label ?? item.title ?? item.name) as string) ?? 'Untitled Source',
      canonicalUrl: ((item.canonicalUrl ?? item.canonical_url ?? item.url) as string) ?? null,
      chunkCount:
        typeof item.chunkCount === 'number' ? item.chunkCount : ((item.chunk_count as number) ?? 0),
      documentCount:
        typeof item.documentCount === 'number'
          ? item.documentCount
          : ((item.document_count as number) ?? 0),
      lastIngestedAt: ((item.lastIngestedAt ?? item.last_ingested_at) as string) ?? null,
      searchText:
        normaliseString(((item.label ?? item.title ?? item.name) as string) ?? '') +
        normaliseString(((item.canonicalUrl ?? item.canonical_url ?? item.url) as string) ?? ''),
    }))
    .filter((source) => Boolean(source.id));

const fetchDatabaseStats = async (): Promise<DatabaseStats> => {
  const empty: DatabaseStats = {
    totalDocuments: 0,
    totalChunks: 0,
    totalSources: 0,
    lastIngestedAt: null,
  };

  try {
    const data = await apiClient.getJson<Record<string, unknown>>('/api/v2/sources/stats');
    if (data) {
      return {
        totalDocuments:
          typeof data.total_documents === 'number'
            ? (data.total_documents as number)
            : ((data.totalDocuments as number) ?? 0),
        totalChunks:
          typeof data.total_chunks === 'number'
            ? (data.total_chunks as number)
            : ((data.totalChunks as number) ?? 0),
        totalSources:
          typeof data.total_sources === 'number'
            ? (data.total_sources as number)
            : ((data.totalSources as number) ?? 0),
        lastIngestedAt:
          typeof data.last_ingested_at === 'string'
            ? (data.last_ingested_at as string)
            : typeof data.lastIngestedAt === 'string'
              ? (data.lastIngestedAt as string)
              : null,
      };
    }
  } catch (error) {
    if (error instanceof ApiError) {
      console.error('Source stats error response:', {
        status: error.status,
        body: error.data,
      });
    } else {
      console.error('Source stats request failed:', error);
    }
  }

  try {
    const countData = await apiClient.getJson<Record<string, unknown>>('/api/v2/sources/count');
    const count = typeof countData.count === 'number' ? countData.count : 0;
    const totalSources =
      typeof countData.total_sources === 'number'
        ? countData.total_sources
        : typeof countData.totalSources === 'number'
          ? countData.totalSources
          : 0;

    return {
      totalDocuments: count,
      totalChunks:
        typeof countData.total_chunks === 'number' ? (countData.total_chunks as number) : count,
      totalSources: (totalSources as number) || count,
      lastIngestedAt: null,
    };
  } catch (error) {
    if (!(error instanceof ApiError)) {
      console.error('Source count request failed:', error);
    }
  }

  // Note: /health?checkRag=true fallback removed - endpoint doesn't exist and causes 404 errors
  return empty;
};

const fetchDatabaseSources = async (): Promise<DatabaseSource[]> => {
  try {
    const payload = await apiClient.getJson<Record<string, unknown>>(
      '/api/v2/sources?page=1&page_size=100',
    );
    if (Array.isArray(payload?.data)) {
      return normaliseSources(payload.data as unknown[]);
    }

    if (Array.isArray(payload?.items)) {
      return normaliseSources(payload.items as unknown[]);
    }

    if (Array.isArray(payload)) {
      return normaliseSources(payload as unknown[]);
    }

    return [];
  } catch (error) {
    console.error('Source fetch failed:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to load sources');
  }
};

export const useDatabasePanel = (
  formatDateDisplay: (value: string | null, includeTime?: boolean) => string | null,
) => {
  const [stats, setStats] = useState<DatabaseStats | null>(null);
  const [sources, setSources] = useState<DatabaseSource[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SourceSort>('date');
  const [isBuildingBM25, setIsBuildingBM25] = useState(false);
  const isRefreshingRef = useRef(false);

  const refreshMetrics = useCallback(async () => {
    // Prevent duplicate in-flight requests
    if (isRefreshingRef.current) {
      return;
    }
    isRefreshingRef.current = true;
    setIsLoading(true);
    setError(null);
    try {
      const [nextStats, nextSources] = await Promise.all([
        fetchDatabaseStats(),
        fetchDatabaseSources(),
      ]);
      setStats(nextStats);
      setSources(nextSources);
    } catch (refreshError) {
      console.error('Failed to refresh database metrics', refreshError);
      setError(
        refreshError instanceof Error ? refreshError.message : 'Failed to load database metrics',
      );
      setSources([]);
    } finally {
      setIsLoading(false);
      isRefreshingRef.current = false;
    }
  }, []);

  const filteredSources = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const base = query
      ? sources.filter((source) => source.searchText.includes(query))
      : [...sources];

    return base.sort(compareBy(sortBy));
  }, [sources, searchQuery, sortBy]);

  const usagePercentage = useMemo(() => {
    if (!stats) return 0;
    return Math.min((stats.totalChunks / MAX_CAPACITY) * 100, 100);
  }, [stats]);

  const lastIngestedLabel = useMemo(
    () => formatDateDisplay(stats?.lastIngestedAt ?? null, true),
    [stats?.lastIngestedAt, formatDateDisplay],
  );

  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value);
  }, []);

  const cycleSourceSort = useCallback(() => {
    setSortBy((current) => (current === 'date' ? 'count' : current === 'count' ? 'name' : 'date'));
  }, []);

  const buildBM25Index = useCallback(async () => {
    setIsBuildingBM25(true);
    try {
      await apiClient.postJson('/api/v2/database/build-bm25', {});
    } catch (e) {
      console.error('BM25 build failed', e);
      throw e;
    } finally {
      setIsBuildingBM25(false);
    }
  }, []);

  return {
    stats,
    sources,
    isLoading,
    error,
    searchQuery,
    sortBy,
    filteredSources,
    usagePercentage,
    lastIngestedLabel,
    setSearchQuery: handleSearchChange,
    cycleSourceSort,
    refreshMetrics,
    buildBM25Index,
    isBuildingBM25,
  };
};
