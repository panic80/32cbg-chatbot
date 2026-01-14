import { ReactNode } from 'react';
import { Zap, FileText, Menu } from 'lucide-react';

export interface WhatsNewUpdateItem {
  icon: ReactNode;
  text: string;
  description?: string;
}

export interface WhatsNewDateGroup {
  date: string; // human readable date label
  updates: WhatsNewUpdateItem[];
}

// Bump this when you add new release notes
export const WHATS_NEW_VERSION = '2025-10-30-rag-refresh';

export const WHATS_NEW_BY_DATE: WhatsNewDateGroup[] = [
  {
    date: 'Thursday, October 30, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Retriever stack restored',
        description:
          'Installed rank_bm25 and updated our LangChain routing so BM25, MMR, multi-query, and unified retrievers are active again—no more silent fallbacks to basic vector search.',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'Clean delegation library',
        description:
          'Purged duplicate Delegation of Authorities PDFs and re-ingested a single canonical copy of each to keep the vector store lean and the top-k results relevant.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Column-aware boost',
        description:
          'Queries that reference column numbers now bubble the exact chunk into context, keeping answers and citations precise.',
      },
    ],
  },
  {
    date: 'Tuesday, October 28, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Streaming-first chat delivery',
        description:
          'Retired the legacy synchronous endpoint and moved the entire chat pipeline to SSE. The UI now talks to one streaming service that handles retrieval, prompting, and token delivery end-to-end.',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'Smarter streams with caching + audits',
        description:
          'Streaming inherits the sync flow’s perks: advanced response caching, stateful retrieval reuse, glossary injections, rich logging, and source audit trails for every answer.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Stable, deterministic retrieval results',
        description:
          'Reworked the result merge to use stable IDs and explicit tie‑breakers across all strategies. Eliminates answer wobble between runs and keeps top sources consistent.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Smart/Fast behavior aligned',
        description:
          'Streaming (Smart mode) now honors glossary definitions the same way as the sync path. Prompts were aligned to treat [Glossary] blocks as authoritative.',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'More consistent model behavior',
        description:
          'Restricted LLM choices to deterministic OpenAI models for RAG prompts to reduce variance in query classification and multi‑query expansion.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Admin Performance Dashboard',
        description: 'Gateway now proxies RAG metrics with caching and optional token.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Faster first token (TTFT)',
        description:
          'Streaming retrieval enabled by default; delayed head streaming tuned; LLM pool warmup and health checks reduce cold starts.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Aligned rate limiting',
        description:
          'Nginx and Express limits aligned with burst support. Added standard X‑RateLimit headers and a Redis-backed shared limiter across processes.',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'Ops hardening',
        description:
          'Production rebuild + reload completed. OS logrotate active for PM2 and RAG logs. pm2‑logrotate migration planned.',
      },
    ],
  },
  {
    date: 'Friday, October 10, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Stateful retrieval with automatic refinement',
        description:
          'Integrated LangGraph with Redis persistence to enable iterative query refinement. Low-quality retrieval now automatically triggers up to 2 refinement cycles, expanding then simplifying queries for better results. Adds ~300ms average overhead but dramatically improves answers for vague questions.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Redis-backed conversation continuity',
        description:
          'All retrieval sessions now persist to Redis with checkpointing at critical workflow nodes. This enables conversation state tracking, better debugging, and provides a complete audit trail for compliance.',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'Smart query optimization strategies',
        description:
          'When retrieval quality is below threshold (avg relevance < 0.4), the system automatically reformulates queries: first by expanding with domain terms and synonyms, then by simplifying to core keywords. Typically improves results from 30% to 70% relevance.',
      },
    ],
  },
  {
    date: 'Sunday, October 5, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'LangGraph caching upgrade',
        description:
          'Hooked every LangGraph prompt into LangChain’s async chat interface and cleaned up caching by serializing documents explicitly. Switched retrieval to the supported aget_relevant_documents path and kept table metadata plus synthesized answers flowing through the new helper.',
      },
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'LangChain backend refresh',
        description:
          'Updated all LangChain imports to the new core modules and repaired the ensemble retriever weights for steadier answers.',
      },
    ],
  },
  {
    date: 'Wednesday, September 17, 2025',
    updates: [
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'Authority matrix and policy ingested',
        description:
          'Both documents are now part of the retrieval set so the chatbot can reference the latest guidance.',
      },
    ],
  },
  {
    date: 'Tuesday, September 16, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Chat replies feel snappier',
        description:
          'Streaming now reuses retrieval pipelines and avoids duplicate placeholders, so answers appear faster and cleaner.',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'New “How this chatbot works” overview',
        description:
          'Explore the retrieval pipeline, guardrails, and why RAG beats a raw LLM inside the new modal in the menu.',
      },
    ],
  },
  {
    date: 'Thursday, August 28, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Embedded FAM - Multi-organization support',
        description:
          'The chatbot now serves CFTDTI, CBI, NJC and FAM with tailored responses for each organization',
      },
    ],
  },
  {
    date: 'Wednesday, August 7, 2025',
    updates: [
      {
        icon: <Zap className="w-4 h-4" />,
        text: 'Implemented Fast (GPT 4) / Smart (GPT 5) toggle',
        description:
          'Use Fast for quick answers and Smart for in-depth, detailed but slower answers',
      },
      {
        icon: <FileText className="w-4 h-4" />,
        text: 'Added Short answer mode',
      },
      {
        icon: <Menu className="w-4 h-4" />,
        text: 'Added a consolidated menu',
      },
    ],
  },
];
