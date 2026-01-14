import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { AnimatedButton } from '@/components/ui/animated-button';
import { Loader2, ChevronDown, ChevronUp, Terminal } from 'lucide-react';
import { InlineIngestionProgress } from '@/components/IngestionProgress';
import IngestionConsole from '@/components/IngestionConsole';
import type { IngestionHistoryEntry } from '../types';

interface IngestionTabProps {
  urlInput: string;
  onUrlChange: (value: string) => void;
  isIngesting: boolean;
  forceRefresh: boolean;
  onForceRefreshChange: (checked: boolean) => void;
  onSubmit: () => void;
  showIngestionProgress: boolean;
  currentIngestionUrl: string;
  progressEndpoint: string | null;
  onProgressComplete: (success: boolean) => void;
  ingestionHistory: IngestionHistoryEntry[];
}

export const IngestionTab: React.FC<IngestionTabProps> = ({
  urlInput,
  onUrlChange,
  isIngesting,
  forceRefresh,
  onForceRefreshChange,
  onSubmit,
  showIngestionProgress,
  currentIngestionUrl,
  progressEndpoint,
  onProgressComplete,
  ingestionHistory,
}) => {
  const [showConsole, setShowConsole] = useState(false);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !isIngesting) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="space-y-4 animate-fade-up">
      <Card className="glass border-border/50">
        <CardHeader>
          <CardTitle>URL Ingestion</CardTitle>
          <CardDescription>
            Add external URLs to the knowledge base. The system will scrape and index the content
            for improved responses.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="url-input">Enter URL to Ingest</Label>
              <div className="flex gap-2">
                <Input
                  id="url-input"
                  type="url"
                  placeholder="https://example.com/document"
                  value={urlInput}
                  onChange={(event) => onUrlChange(event.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={isIngesting}
                />
                <AnimatedButton
                  onClick={onSubmit}
                  disabled={isIngesting || !urlInput.trim()}
                  ripple
                >
                  {isIngesting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Ingesting...
                    </>
                  ) : (
                    'Ingest URL'
                  )}
                </AnimatedButton>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="force-refresh"
                  checked={forceRefresh}
                  onCheckedChange={(checked) => onForceRefreshChange(Boolean(checked))}
                />
                <Label htmlFor="force-refresh" className="text-sm font-normal cursor-pointer">
                  Force refresh (re-ingest even if document already exists)
                </Label>
              </div>
              <p className="text-xs text-muted-foreground">
                The content will be automatically processed and added to the RAG knowledge base.
              </p>
            </div>

            {showIngestionProgress && currentIngestionUrl && progressEndpoint && (
              <div className="mt-4 space-y-3">
                <InlineIngestionProgress
                  url={currentIngestionUrl}
                  progressEndpoint={progressEndpoint}
                  onComplete={onProgressComplete}
                />

                {/* Collapsible Console */}
                <div className="border rounded-lg overflow-hidden">
                  <button
                    onClick={() => setShowConsole(!showConsole)}
                    className="w-full flex items-center justify-between px-4 py-2 bg-muted/50 hover:bg-muted transition-colors text-sm"
                  >
                    <div className="flex items-center gap-2">
                      <Terminal className="h-4 w-4" />
                      <span>Console Log</span>
                    </div>
                    {showConsole ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                  {showConsole && (
                    <IngestionConsole
                      url={currentIngestionUrl}
                      progressEndpoint={progressEndpoint}
                      className="border-0 rounded-none"
                    />
                  )}
                </div>
              </div>
            )}

            {ingestionHistory.length > 0 && !showIngestionProgress && (
              <div className="space-y-2">
                <Label>Recent Ingestions</Label>
                <div className="space-y-2">
                  {ingestionHistory.map((entry, index) => (
                    <div
                      key={`${entry.url}-${index}`}
                      className={`p-3 rounded-lg border ${
                        entry.status === 'success'
                          ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950'
                          : entry.status === 'exists'
                            ? 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950'
                            : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{entry.url}</p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(entry.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            entry.status === 'success'
                              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                              : entry.status === 'exists'
                                ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                          }`}
                        >
                          {entry.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <h4 className="text-sm font-medium mb-2">Tips for URL Ingestion:</h4>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>• Make sure the URL is publicly accessible</li>
              <li>• The system will extract text content from web pages</li>
              <li>• PDF documents and other file types may be supported depending on the URL</li>
              <li>• Large documents will be split into smaller chunks for processing</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
