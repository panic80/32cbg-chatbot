export type Effect = 'allow' | 'deny' | 'require' | 'limit' | 'n/a';

export type ChangeType =
  | 'stricter'
  | 'looser'
  | 'additionalRequirement'
  | 'exception'
  | 'replacement'
  | 'notApplicable'
  | 'addition';

export interface CitationRef {
  sourceId: string;
  anchor?: string;
}

export interface PolicyUnit {
  policyArea: string;
  dedupeKey: string;
  subject: string;
  action: string;
  conditions: string[];
  effect: Effect;
  scope?: string;
  notes?: string;
  citations: CitationRef[];
  audience?: 'general' | 'classA' | string;
}

export interface DeltaItem {
  policyArea: string;
  dedupeKey: string;
  changeType: ChangeType;
  summary: string;
  citations: string[]; // sourceIds
  baseline?: PolicyUnit;
  classA?: PolicyUnit;
}

export interface DeltaResponse {
  stricter: DeltaItem[];
  looser: DeltaItem[];
  additionalRequirements: DeltaItem[];
  exceptions: DeltaItem[];
  replacements: DeltaItem[];
  notApplicable: DeltaItem[];
  additions: DeltaItem[];
  debug?: Record<string, unknown>;
}
