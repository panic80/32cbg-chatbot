/**
 * Elite Chat Interface - Type Definitions
 */

export interface FollowUpQuestion {
  id: string;
  question: string;
  category?: 'clarification' | 'related' | 'practical' | 'explore' | 'general';
  icon?: string;
  confidence?: number;
  sourceGrounding?: string;
  groundingScore?: number;
}

export interface Source {
  id: string;
  text: string;
  title?: string;
  url?: string;
  section?: string;
  page?: number;
  score?: number;
  reference?: string;
  metadata?: {
    type?: string;
    last_modified?: string;
    tags?: string[];
    [key: string]: unknown;
  };
}

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: number;
  status?: 'sending' | 'sent' | 'delivered' | 'read' | 'error';
  isTyping?: boolean;
  attachments?: Attachment[];
  reactions?: Reaction[];
  metadata?: Record<string, unknown>;
  sources?: Source[];
  isFormatted?: boolean;
  followUpQuestions?: FollowUpQuestion[];
  modelMode?: 'fast' | 'smart';
  shortAnswerMode?: boolean;
  // Optional audience-specific policy differences payload
  delta?: import('./policy').DeltaResponse;
}

export interface Attachment {
  id: string;
  type: 'image' | 'document' | 'video' | 'link';
  url: string;
  name?: string;
  size?: number;
  thumbnail?: string;
  metadata?: Record<string, unknown>;
}

export interface Reaction {
  id: string;
  emoji: string;
  count: number;
  users: string[];
}

export interface ChatOptions {
  showAvatars?: boolean;
  enableReactions?: boolean;
  enableAttachments?: boolean;
  messageLimit?: number;
  autoScroll?: boolean;
  typingIndicatorTimeout?: number;
}

export interface User {
  id: string;
  name: string;
  avatar?: string;
  status?: 'online' | 'away' | 'offline';
  isTyping?: boolean;
}

export type Theme = 'light' | 'dark';
