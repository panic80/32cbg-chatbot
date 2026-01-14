import React from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle, 
  Button, 
  Input, 
  Label, 
  Skeleton 
} from '@/components/ui';
import { Loader2, RefreshCw, TrendingUp } from 'lucide-react';
import type { ChatLogEntry, LogFilters } from '../types';
import type { VisitSummary } from '@/api/analytics';

type LogsPagination = {
  limit: number;
  offset: number;
  hasMore: boolean;
  nextOffset: number | null;
};

interface LogsTabProps {
  visitSummary: VisitSummary | null;
  visitSummaryError: string | null;
  visitSummaryLoading: boolean;
  onRefreshVisitSummary: () => void;
  visitDailyCounts: Array<{ date: string; count: number }>;
  logsFilters: LogFilters;
  onFiltersChange: React.Dispatch<React.SetStateAction<LogFilters>>;
  logsLoading: boolean;
  logsError: string | null;
  chatLogs: ChatLogEntry[];
  formatDateDisplay: (value: string | null, includeTime?: boolean) => string | null;
  formatBooleanLabel: (value: boolean | null) => string;
  summariseMetadata: (metadata: unknown) => string | null;
  logsPagination: LogsPagination;
  onApplyFilters: () => void;
  onResetFilters: () => void;
  onRefreshLogs: () => void;
  onNextPage: () => void;
  onPreviousPage: () => void;
}

export const LogsTab: React.FC<LogsTabProps> = ({
  visitSummary,
  visitSummaryError,
  visitSummaryLoading,
  onRefreshVisitSummary,
  visitDailyCounts,
  logsFilters,
  onFiltersChange,
  logsLoading,
  logsError,
  chatLogs,
  formatDateDisplay,
  formatBooleanLabel,
  summariseMetadata,
  logsPagination,
  onApplyFilters,
  onResetFilters,
  onRefreshLogs,
  onNextPage,
  onPreviousPage,
}) => {
  return (
    <div className="space-y-4 animate-fade-up">
      <Card className="glass border-border/50">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Visit Analytics
            </CardTitle>
            <CardDescription>
              High-level page view insights captured from the visit logger.
            </CardDescription>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onRefreshVisitSummary}
            disabled={visitSummaryLoading}
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${visitSummaryLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </CardHeader>
        <CardContent className="space-y-6">
          {visitSummaryError ? (
            <div className="rounded-md border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800 dark:border-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200">
              {visitSummaryError}
            </div>
          ) : visitSummaryLoading && !visitSummary ? (
            <div className="grid gap-4 md:grid-cols-3">
              {[0, 1, 2].map((index) => (
                <Skeleton key={index} className="h-24 rounded-lg" />
              ))}
              <Skeleton className="md:col-span-3 h-40 rounded-lg" />
            </div>
          ) : (
            <>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-lg border border-border/60 bg-background/60 p-4 shadow-sm">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    Total visits
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-foreground">
                    {(visitSummary?.totalVisits ?? 0).toLocaleString()}
                  </p>
                </div>
                <div className="rounded-lg border border-border/60 bg-background/60 p-4 shadow-sm">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    First recorded visit
                  </p>
                  <p className="mt-2 text-sm font-medium text-foreground">
                    {formatDateDisplay(visitSummary?.firstVisit ?? null, true) ?? '—'}
                  </p>
                </div>
                <div className="rounded-lg border border-border/60 bg-background/60 p-4 shadow-sm">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    Most recent visit
                  </p>
                  <p className="mt-2 text-sm font-medium text-foreground">
                    {formatDateDisplay(visitSummary?.lastVisit ?? null, true) ?? '—'}
                  </p>
                </div>
              </div>

              <div className="rounded-lg border border-border/60 bg-background/60 p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-foreground">Last 7 days</p>
                  <span className="text-xs text-muted-foreground">
                    {visitDailyCounts.length === 0 ? 'No visit data yet' : 'Daily totals'}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {visitDailyCounts.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      We will show daily visit counts once traffic arrives.
                    </p>
                  ) : (
                    visitDailyCounts.map((entry) => {
                      const safeDate = entry.date ? new Date(`${entry.date}T00:00:00Z`) : null;
                      const displayDate =
                        safeDate && !Number.isNaN(safeDate.getTime())
                          ? safeDate.toLocaleDateString()
                          : entry.date;
                      return (
                        <div
                          key={entry.date}
                          className="flex items-center justify-between rounded-md bg-muted/40 px-3 py-2 text-sm"
                        >
                          <span>{displayDate}</span>
                          <span className="font-medium text-foreground">
                            {entry.count.toLocaleString()}
                          </span>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle>Chat Analytics Logs</CardTitle>
          <CardDescription>
            Filter and review chat interactions captured by the gateway. Logs are limited to recent
            activity for performance.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="log-search">Search questions</Label>
              <Input
                id="log-search"
                placeholder="Meal allowances in Ottawa"
                value={logsFilters.search}
                onChange={(event) =>
                  onFiltersChange((prev) => ({ ...prev, search: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-conversation">Conversation ID</Label>
              <Input
                id="log-conversation"
                placeholder="conv_123"
                value={logsFilters.conversationId}
                onChange={(event) =>
                  onFiltersChange((prev) => ({ ...prev, conversationId: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-provider">Provider</Label>
              <select
                id="log-provider"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                value={logsFilters.provider}
                onChange={(event) =>
                  onFiltersChange((prev) => ({ ...prev, provider: event.target.value }))
                }
              >
                <option value="all">All providers</option>
                <option value="openai">OpenAI</option>
                <option value="google">Google</option>
                <option value="anthropic">Anthropic</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-model">Model</Label>
              <Input
                id="log-model"
                placeholder="gpt-5-mini"
                value={logsFilters.model}
                onChange={(event) =>
                  onFiltersChange((prev) => ({ ...prev, model: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-start">Start date</Label>
              <Input
                id="log-start"
                type="date"
                value={logsFilters.startAt}
                onChange={(event) =>
                  onFiltersChange((prev) => ({ ...prev, startAt: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-end">End date</Label>
              <Input
                id="log-end"
                type="date"
                value={logsFilters.endAt}
                onChange={(event) =>
                  onFiltersChange((prev) => ({ ...prev, endAt: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-rag">RAG enabled</Label>
              <select
                id="log-rag"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                value={logsFilters.ragEnabled}
                onChange={(event) =>
                  onFiltersChange((prev) => ({
                    ...prev,
                    ragEnabled: event.target.value as LogFilters['ragEnabled'],
                  }))
                }
              >
                <option value="all">All</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="log-short">Short answer mode</Label>
              <select
                id="log-short"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                value={logsFilters.shortAnswerMode}
                onChange={(event) =>
                  onFiltersChange((prev) => ({
                    ...prev,
                    shortAnswerMode: event.target.value as LogFilters['shortAnswerMode'],
                  }))
                }
              >
                <option value="all">All</option>
                <option value="true">Enabled</option>
                <option value="false">Disabled</option>
              </select>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={onApplyFilters} disabled={logsLoading}>
              {logsLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Loading
                </>
              ) : (
                'Apply filters'
              )}
            </Button>
            <Button type="button" variant="outline" onClick={onResetFilters} disabled={logsLoading}>
              Reset
            </Button>
            <Button type="button" variant="ghost" onClick={onRefreshLogs} disabled={logsLoading}>
              <RefreshCw className={`mr-2 h-4 w-4 ${logsLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <span className="text-xs text-muted-foreground ml-auto">
              Showing {chatLogs.length} of{' '}
              {logsPagination.hasMore ? `${logsPagination.limit}+` : chatLogs.length} results
            </span>
          </div>

          {logsError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
              {logsError}
            </div>
          )}

          <div className="space-y-4">
            {logsLoading ? (
              <div className="space-y-3">
                {[0, 1, 2].map((index) => (
                  <Skeleton key={index} className="h-32 rounded-lg" />
                ))}
              </div>
            ) : chatLogs.length === 0 ? (
              <div className="rounded-lg border border-border/60 bg-muted/30 px-4 py-6 text-center text-sm text-muted-foreground">
                No chat activity found for the selected filters.
              </div>
            ) : (
              <div className="space-y-3">
                {chatLogs.map((log) => {
                  const askedAtLabel = formatDateDisplay(log.askedAt, true) ?? 'Unknown time';
                  const metadataSummary = summariseMetadata(log.metadata);
                  const answerPreview = log.answer
                    ? log.answer.length > 280
                      ? `${log.answer.slice(0, 277)}…`
                      : log.answer
                    : null;
                  return (
                    <div
                      key={log.id}
                      className="rounded-lg border border-border/60 bg-background/60 p-4 shadow-sm"
                    >
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">{askedAtLabel}</span>
                        {log.provider && (
                          <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] uppercase tracking-wide text-foreground/80">
                            {log.provider}
                          </span>
                        )}
                        {log.model && (
                          <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-foreground/80">
                            {log.model}
                          </span>
                        )}
                        <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-foreground/70">
                          RAG: {formatBooleanLabel(log.ragEnabled)}
                        </span>
                        <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] text-foreground/70">
                          Short answers: {formatBooleanLabel(log.shortAnswerMode)}
                        </span>
                        {log.conversationId && (
                          <span
                            className="truncate text-[11px] text-muted-foreground"
                            title={log.conversationId}
                          >
                            Conversation: {log.conversationId}
                          </span>
                        )}
                      </div>
                      <div className="mt-3 space-y-2">
                        <div>
                          <p className="text-sm font-semibold text-foreground">Question</p>
                          <p className="text-sm text-muted-foreground whitespace-pre-line">
                            {log.question}
                          </p>
                        </div>
                        {answerPreview && (
                          <div>
                            <p className="text-sm font-semibold text-foreground">Answer</p>
                            <p className="text-sm text-muted-foreground whitespace-pre-line">
                              {answerPreview}
                            </p>
                          </div>
                        )}
                        {metadataSummary && (
                          <div className="text-xs text-muted-foreground">
                            Metadata: {metadataSummary}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={onPreviousPage}
              disabled={logsLoading || logsPagination.offset <= 0}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onNextPage}
              disabled={
                logsLoading || !logsPagination.hasMore || logsPagination.nextOffset === null
              }
            >
              Next
            </Button>
            <span className="text-xs text-muted-foreground">
              Page {logsPagination.offset / logsPagination.limit + 1}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
