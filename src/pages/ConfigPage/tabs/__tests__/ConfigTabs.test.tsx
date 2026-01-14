import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ModelSettingsTab } from '../ModelSettingsTab';
import { IngestionTab } from '../IngestionTab';
import { DatabaseTab } from '../DatabaseTab';
import { LogsTab } from '../LogsTab';
import type { LLMModel } from '@/constants/models';
import type { VisitSummary } from '@/api/analytics';
import {
  LOGS_FILTER_DEFAULTS,
  type DatabaseStats,
  type DatabaseSource,
  type LogFilters,
  type IngestionHistoryEntry,
} from '../../types';

const models: LLMModel[] = [
  {
    id: 'openai-model',
    name: 'OpenAI Model',
    provider: 'openai',
    description: 'OpenAI test model',
  },
  {
    id: 'google-model',
    name: 'Google Model',
    provider: 'google',
    description: 'Google test model',
  },
];

describe('Config tabs', () => {
  it('renders model options and responds to selection changes', () => {
    const handleProviderChange = vi.fn();
    const handleModelChange = vi.fn();
    const handleSave = vi.fn();
    const handleReset = vi.fn();

    render(
      <ModelSettingsTab
        models={models}
        selectedModel="openai-model"
        tempSelectedModel="openai-model"
        tempSelectedProvider="openai"
        hasUnsavedChanges
        onProviderChange={handleProviderChange}
        onModelChange={handleModelChange}
        onSave={handleSave}
        onReset={handleReset}
      />,
    );

    const modelTiles = screen.getAllByText('OpenAI Model');
    expect(modelTiles.length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole('tab', { name: /^Google$/i }));

    fireEvent.click(modelTiles[0]);
    expect(handleModelChange).toHaveBeenCalledWith('openai-model');
  });

  it('renders ingestion history and triggers callbacks', () => {
    const onUrlChange = vi.fn();
    const onForceRefreshChange = vi.fn();
    const onSubmit = vi.fn();
    const onProgressComplete = vi.fn();

    const historyEntry: IngestionHistoryEntry = {
      url: 'https://example.com',
      status: 'success',
      timestamp: new Date().toISOString(),
    };

    render(
      <IngestionTab
        urlInput="https://example.com"
        onUrlChange={onUrlChange}
        isIngesting={false}
        forceRefresh={false}
        onForceRefreshChange={onForceRefreshChange}
        onSubmit={onSubmit}
        showIngestionProgress={false}
        currentIngestionUrl=""
        progressEndpoint={null}
        onProgressComplete={onProgressComplete}
        ingestionHistory={[historyEntry]}
      />,
    );

    expect(screen.getByText(historyEntry.url)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /ingest url/i }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it('shows database stats and invokes action handlers', () => {
    const stats: DatabaseStats = {
      totalDocuments: 10,
      totalChunks: 20,
      totalSources: 5,
      lastIngestedAt: new Date().toISOString(),
    };

    const sources: DatabaseSource[] = [
      {
        id: 'source-1',
        label: 'Example Source',
        canonicalUrl: 'https://example.com',
        chunkCount: 3,
        documentCount: 2,
        lastIngestedAt: new Date().toISOString(),
        searchText: 'example source',
      },
    ];

    const onExport = vi.fn();
    const onRefresh = vi.fn();
    const onCycleSort = vi.fn();
    const onSearchChange = vi.fn();
    const onPurge = vi.fn();
    const onToggleLog = vi.fn();

    render(
      <DatabaseTab
        stats={stats}
        usagePercentage={50}
        lastIngestedLabel={'Today'}
        isLoading={false}
        onExport={onExport}
        onRefresh={onRefresh}
        sources={sources}
        filteredSources={sources}
        sourceSearchQuery=""
        onSourceSearchQueryChange={onSearchChange}
        sourceSortBy="date"
        onCycleSourceSort={onCycleSort}
        formatDateDisplay={(value) => value}
        sourcesError={null}
        isPurging={false}
        onPurge={onPurge}
        activityLog={[]}
        showActivityLog={false}
        onToggleActivityLog={onToggleLog}
      />,
    );

    const overviewCard = screen.getByText('Database Overview').closest('div');
    expect(overviewCard).toBeTruthy();
    const buttons = within(overviewCard as HTMLElement).getAllByRole('button');
    fireEvent.click(buttons[1]);
    expect(onRefresh).toHaveBeenCalledTimes(1);

    const searchInput = screen.getByPlaceholderText(/search sources/i);
    fireEvent.change(searchInput, { target: { value: 'query' } });
    expect(onSearchChange).toHaveBeenCalled();
  });

  it('handles logs filter actions', () => {
    const visitSummary = {
      totalVisits: 3,
      firstVisit: new Date().toISOString(),
      lastVisit: new Date().toISOString(),
      dailyCounts: [{ date: '2025-01-01', count: 1 }],
    } as VisitSummary;

    const logsFilters: LogFilters = { ...LOGS_FILTER_DEFAULTS };
    const logsPagination = {
      limit: 20,
      offset: 20,
      hasMore: true,
      nextOffset: 40,
    } as const;

    const setFilters = vi.fn();
    const onApplyFilters = vi.fn();
    const onResetFilters = vi.fn();
    const onRefreshLogs = vi.fn();
    const onNextPage = vi.fn();
    const onPreviousPage = vi.fn();

    render(
      <LogsTab
        visitSummary={visitSummary}
        visitSummaryError={null}
        visitSummaryLoading={false}
        onRefreshVisitSummary={vi.fn()}
        visitDailyCounts={visitSummary.dailyCounts}
        logsFilters={logsFilters}
        onFiltersChange={setFilters as unknown as React.Dispatch<React.SetStateAction<LogFilters>>}
        logsLoading={false}
        logsError={null}
        chatLogs={[]}
        formatDateDisplay={(value) => value}
        formatBooleanLabel={(value) => (value ? 'Yes' : 'No')}
        summariseMetadata={() => 'meta'}
        logsPagination={logsPagination}
        onApplyFilters={onApplyFilters}
        onResetFilters={onResetFilters}
        onRefreshLogs={onRefreshLogs}
        onNextPage={onNextPage}
        onPreviousPage={onPreviousPage}
      />,
    );

    fireEvent.change(screen.getByLabelText(/search questions/i), { target: { value: 'meal' } });
    expect(setFilters).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: /apply filters/i }));
    expect(onApplyFilters).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: /previous/i }));
    expect(onPreviousPage).toHaveBeenCalled();
  });
});
