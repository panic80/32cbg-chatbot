import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  AnimatedButton,
  Button,
  Input,
  Progress,
  Skeleton,
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui';
import {
  Activity,
  Clock,
  Database,
  Download,
  FileText,
  Filter,
  Globe,
  HardDrive,
  Hash,
  Loader2,
  RefreshCw,
  Search,
  Trash2,
  TrendingUp,
} from 'lucide-react';
import type { ActivityLogEntry, DatabaseSource, DatabaseStats } from '../types';

type SourceSort = 'date' | 'count' | 'name';

type DatabaseTabProps = {
  stats: DatabaseStats | null;
  usagePercentage: number;
  lastIngestedLabel: string | null;
  isLoading: boolean;
  onExport: () => void;
  onRefresh: () => void;
  sources: DatabaseSource[];
  filteredSources: DatabaseSource[];
  sourceSearchQuery: string;
  onSourceSearchQueryChange: (value: string) => void;
  sourceSortBy: SourceSort;
  onCycleSourceSort: () => void;
  formatDateDisplay: (value: string | null, includeTime?: boolean) => string | null;
  sourcesError: string | null;
  isPurging: boolean;
  onPurge: () => void;
  activityLog: ActivityLogEntry[];
  showActivityLog: boolean;
  onToggleActivityLog: () => void;
  onBuildBM25: () => void;
  isBuildingBM25: boolean;
};

export const DatabaseTab: React.FC<DatabaseTabProps> = ({
  stats,
  usagePercentage,
  lastIngestedLabel,
  isLoading,
  onExport,
  onRefresh,
  sources,
  filteredSources,
  sourceSearchQuery,
  onSourceSearchQueryChange,
  sourceSortBy,
  onCycleSourceSort,
  formatDateDisplay,
  sourcesError,
  isPurging,
  onPurge,
  activityLog,
  showActivityLog,
  onToggleActivityLog,
  onBuildBM25,
  isBuildingBM25,
}) => {
  return (
    <div className="space-y-4 animate-fade-up">
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle>Database Management</CardTitle>
          <CardDescription>
            Manage the RAG (Retrieval-Augmented Generation) vector database that stores indexed
            documents.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  Database Overview
                </h4>
                <div className="flex gap-2">
                  <AnimatedButton
                    variant="ghost"
                    size="sm"
                    onClick={onExport}
                    disabled={!stats || isLoading}
                    ripple
                  >
                    <Download className="h-4 w-4" />
                  </AnimatedButton>
                  <AnimatedButton
                    variant="ghost"
                    size="sm"
                    onClick={onRefresh}
                    disabled={isLoading}
                    ripple
                  >
                    <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  </AnimatedButton>
                </div>
              </div>

              {isLoading ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <Skeleton className="h-24 rounded-lg" />
                    <Skeleton className="h-24 rounded-lg" />
                    <Skeleton className="h-24 rounded-lg" />
                  </div>
                  <Skeleton className="h-20 rounded-lg" />
                </div>
              ) : stats ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-muted/50 rounded-lg glass-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <FileText className="h-5 w-5 text-muted-foreground" />
                        <TrendingUp className="h-4 w-4 text-green-500" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Total Documents</p>
                        <p className="text-2xl font-bold animate-scale-in">
                          {stats.totalDocuments.toLocaleString()}
                        </p>
                      </div>
                    </div>

                    <div className="p-4 bg-muted/50 rounded-lg glass-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <Hash className="h-5 w-5 text-muted-foreground" />
                        <Activity className="h-4 w-4 text-blue-500" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Total Chunks</p>
                        <p className="text-2xl font-bold animate-scale-in">
                          {stats.totalChunks.toLocaleString()}
                        </p>
                      </div>
                    </div>

                    <div className="p-4 bg-muted/50 rounded-lg glass-sm space-y-3">
                      <div className="flex items-center justify-between">
                        <HardDrive className="h-5 w-5 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground">
                          {usagePercentage.toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-2">Storage Usage</p>
                        <Progress value={usagePercentage} className="h-2" />
                      </div>
                    </div>
                  </div>

                  <div className="p-3 bg-muted/30 rounded-lg space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Total Sources</span>
                      <span className="font-semibold">{stats.totalSources.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Avg chunks per doc</span>
                      <span className="font-semibold">
                        {stats.totalDocuments > 0
                          ? (stats.totalChunks / stats.totalDocuments).toFixed(1)
                          : '0'}
                      </span>
                    </div>
                    {lastIngestedLabel && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Last ingestion:</span>
                        <span className="font-medium">{lastIngestedLabel}</span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-4">
                  Unable to load database statistics
                </p>
              )}
            </div>

            {!isLoading &&
              (sources.length > 0 ? (
                <div className="p-4 border rounded-lg space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium">Indexed Sources</h4>
                    <div className="flex items-center gap-2">
                      <div className="relative">
                        <Search className="h-4 w-4 absolute left-2 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                        <Input
                          type="text"
                          placeholder="Search sources..."
                          value={sourceSearchQuery}
                          onChange={(event) => onSourceSearchQueryChange(event.target.value)}
                          className="h-8 pl-8 pr-3 text-xs w-48"
                        />
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={onCycleSourceSort}
                        className="text-xs"
                      >
                        <Filter className="h-3 w-3 mr-1" />
                        {sourceSortBy === 'date'
                          ? 'Date'
                          : sourceSortBy === 'count'
                            ? 'Count'
                            : 'Name'}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {filteredSources.length > 0 ? (
                      filteredSources.map((source) => {
                        const displayUrl = source.canonicalUrl
                          ? source.canonicalUrl.replace(/^https?:\/\//, '').replace(/\/$/, '')
                          : null;
                        const lastIndexedLabel = formatDateDisplay(source.lastIngestedAt);

                        return (
                          <div
                            key={source.id}
                            className="p-3 bg-muted/50 rounded-lg hover:bg-muted/70 transition-colors"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1 min-w-0 space-y-1">
                                <div className="flex items-center gap-2">
                                  <Globe className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                                  <span className="font-medium text-sm truncate">
                                    {source.label}
                                  </span>
                                </div>
                                {displayUrl && (
                                  <p className="text-xs text-muted-foreground truncate pl-6">
                                    {displayUrl}
                                  </p>
                                )}
                                <div className="flex items-center gap-4 text-xs text-muted-foreground pl-6">
                                  <span className="flex items-center gap-1">
                                    <FileText className="h-3 w-3" />
                                    {source.documentCount} docs
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <Hash className="h-3 w-3" />
                                    {source.chunkCount} chunks
                                  </span>
                                  <span className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {lastIndexedLabel ?? 'Unknown'}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <p className="text-xs text-muted-foreground text-center py-2">
                        No sources match your search
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <p
                  className={`text-xs text-center py-4 ${
                    sourcesError ? 'text-destructive' : 'text-muted-foreground'
                  }`}
                >
                  {sourcesError ??
                    'No sources have been indexed yet. Ingest content to populate this list.'}
                </p>
              ))}

            <div className="p-4 border rounded-lg space-y-4">
              <div className="space-y-1">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-blue-500" />
                  Optimization
                </h4>
                <p className="text-xs text-muted-foreground">
                  Perform maintenance tasks to optimize retrieval performance.
                </p>
              </div>

              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="space-y-1">
                  <div className="text-sm font-medium">BM25 Index</div>
                  <div className="text-xs text-muted-foreground">
                    Rebuilds the keyword search index for hybrid retrieval.
                  </div>
                </div>
                <AnimatedButton
                  variant="outline"
                  size="sm"
                  onClick={onBuildBM25}
                  disabled={isBuildingBM25}
                  ripple
                >
                  {isBuildingBM25 ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Building...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Rebuild Index
                    </>
                  )}
                </AnimatedButton>
              </div>
            </div>

            <div className="p-4 border rounded-lg space-y-4">
              <div className="space-y-1">
                <h4 className="text-sm font-medium text-destructive">Danger Zone</h4>
                <p className="text-xs text-muted-foreground">
                  Purging the database removes all indexed documents and embeddings. This cannot be
                  undone.
                </p>
              </div>

              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <AnimatedButton variant="destructive" size="sm" disabled={isPurging} ripple>
                    {isPurging ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Purging...
                      </>
                    ) : (
                      <>
                        <Trash2 className="mr-2 h-4 w-4" />
                        Purge Database
                      </>
                    )}
                  </AnimatedButton>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. This will permanently delete all indexed
                      documents from the vector database, including:
                      <ul className="mt-2 space-y-1 list-disc list-inside">
                        <li>All ingested URLs and their content</li>
                        <li>All document embeddings and metadata</li>
                        <li>All conversation history references</li>
                      </ul>
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={onPurge}
                      className="bg-red-600 hover:bg-red-700 text-white"
                    >
                      Yes, purge database
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>

            {activityLog.length > 0 && (
              <div className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <Activity className="h-4 w-4" />
                    Recent Activity
                  </h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onToggleActivityLog}
                    className="text-xs"
                  >
                    {showActivityLog ? 'Hide' : 'Show'} Log
                  </Button>
                </div>

                {showActivityLog && (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {activityLog.slice(0, 10).map((entry, index) => (
                      <div
                        key={`${entry.timestamp}-${index}`}
                        className="flex items-start gap-2 text-xs"
                      >
                        <Clock className="h-3 w-3 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{entry.action}</span>
                            <span className="text-muted-foreground">
                              {new Date(entry.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <p className="text-muted-foreground">{entry.details}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="p-4 bg-muted rounded-lg">
              <h4 className="text-sm font-medium mb-2">Database Information:</h4>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• The database stores document embeddings for semantic search</li>
                <li>• Purging will not affect your chat history</li>
                <li>• You can re-ingest documents at any time</li>
                <li>• The database uses ChromaDB for vector storage</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
