import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import {
  FileText,
  Globe,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  BookOpen,
  Calendar,
  Hash,
} from 'lucide-react';
import { Source } from '@/types/chat';

interface SourcesDisplayProps {
  sources: Source[];
  className?: string;
}

export const SourcesDisplay: React.FC<SourcesDisplayProps> = ({ sources, className = '' }) => {
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState(false);

  // Initialize all sources as expanded by default
  useEffect(() => {
    if (sources && sources.length > 0) {
      const allSourceIds = new Set(sources.map((source) => source.id));
      setExpandedSources(allSourceIds);
    }
  }, [sources]);

  if (!sources || sources.length === 0) {
    return null;
  }

  const toggleSource = (sourceId: string) => {
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(sourceId)) {
      newExpanded.delete(sourceId);
    } else {
      newExpanded.add(sourceId);
    }
    setExpandedSources(newExpanded);
  };

  const getSourceIcon = (source: Source) => {
    const type = source.metadata?.type;
    if (type === 'pdf') return <FileText className="w-4 h-4" />;
    if (type === 'web') return <Globe className="w-4 h-4" />;
    return <BookOpen className="w-4 h-4" />;
  };

  const getSourceTypeColor = (source: Source) => {
    const type = source.metadata?.type;
    if (type === 'pdf') return 'destructive';
    if (type === 'web') return 'default';
    return 'secondary';
  };

  const displayedSources = showAll ? sources : sources.slice(0, 3);

  return (
    <div className={`sources-display ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-muted-foreground flex items-center gap-2">
          <BookOpen className="w-4 h-4" />
          Sources ({sources.length})
        </h3>
        {sources.length > 3 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll(!showAll)}
            className="text-xs"
          >
            {showAll ? 'Show Less' : `Show All (${sources.length})`}
          </Button>
        )}
      </div>

      <ScrollArea className={showAll ? 'h-[400px]' : ''}>
        <div className="space-y-2">
          {displayedSources.map((source, index) => (
            <Card key={source.id} className="overflow-hidden">
              <Collapsible
                open={expandedSources.has(source.id)}
                onOpenChange={() => toggleSource(source.id)}
              >
                <CollapsibleTrigger className="w-full">
                  <CardContent className="p-3 cursor-pointer hover:bg-muted/50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          {getSourceIcon(source)}
                          <span className="font-medium text-sm">
                            {source.title || `Source ${index + 1}`}
                          </span>
                          {source.score && (
                            <Badge variant="outline" className="text-xs">
                              {Math.round(source.score * 100)}% match
                            </Badge>
                          )}
                          <Badge variant={getSourceTypeColor(source)} className="text-xs">
                            {source.metadata?.type || 'document'}
                          </Badge>
                        </div>

                        <p className="text-xs text-muted-foreground line-clamp-2">{source.text}</p>

                        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                          {source.section && (
                            <span className="flex items-center gap-1">
                              <Hash className="w-3 h-3" />
                              {source.section}
                            </span>
                          )}
                          {source.page && (
                            <span className="flex items-center gap-1">
                              <FileText className="w-3 h-3" />
                              Page {source.page}
                            </span>
                          )}
                          {source.metadata?.last_modified && (
                            <span className="flex items-center gap-1">
                              <Calendar className="w-3 h-3" />
                              {new Date(source.metadata.last_modified).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="ml-2">
                        {expandedSources.has(source.id) ? (
                          <ChevronUp className="w-4 h-4 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </CardContent>
                </CollapsibleTrigger>

                <CollapsibleContent>
                  <div className="px-3 pb-3 pt-0">
                    <div className="bg-muted/30 rounded p-3 text-sm">
                      <p className="whitespace-pre-wrap">{source.text}</p>
                    </div>

                    {source.url && (
                      <div className="mt-3">
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                        >
                          View Original Source
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    )}

                    {source.metadata?.tags && source.metadata.tags.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {source.metadata.tags.map((tag: string) => (
                          <Badge key={tag} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};
