import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ensureDirectory = (dirPath: string) => {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
};

const resolveDatabasePath = (): string => {
  if (process.env.CHAT_LOG_DB_PATH) {
    return path.resolve(process.env.CHAT_LOG_DB_PATH);
  }

  if (process.env.NODE_ENV === 'test') {
    const testDir = process.env.CHAT_LOG_TEST_DIR
      ? path.resolve(process.env.CHAT_LOG_TEST_DIR)
      : os.tmpdir();
    ensureDirectory(testDir);
    return path.join(testDir, `cbthis-chat-logs-${process.pid}.sqlite`);
  }

  return path.join(__dirname, '..', 'data', 'analytics.sqlite');
};

const toBooleanFlag = (value: unknown): number | null => {
  if (typeof value === 'boolean') {
    return value ? 1 : 0;
  }
  if (value === 1 || value === 0) {
    return value;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1') {
      return 1;
    }
    if (normalized === 'false' || normalized === '0') {
      return 0;
    }
  }
  return null;
};

const toMetadataString = (metadata: unknown): string | null => {
  if (!metadata) return null;
  try {
    return typeof metadata === 'string' ? metadata : JSON.stringify(metadata);
  } catch (error) {
    return null;
  }
};

const compactObject = (input: unknown): Record<string, unknown> | null => {
  if (!input || typeof input !== 'object') {
    return null;
  }

  const entries = Object.entries(input as Record<string, unknown>).filter(([, value]) => {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') {
      return value.trim().length > 0;
    }
    return true;
  });

  if (entries.length === 0) {
    return null;
  }

  return Object.fromEntries(entries);
};

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVELS: LogLevel[] = ['debug', 'info', 'warn', 'error'];
const LEVEL_RANK: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

const normalizeLevel = (value: unknown): LogLevel => {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  return LOG_LEVELS.includes(normalized as LogLevel) ? (normalized as LogLevel) : 'info';
};

interface SerializedError {
  name?: string;
  message?: string;
  stack?: string;
  [key: string]: unknown;
}

const serializeError = (error: unknown): SerializedError | unknown | null => {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }
  if (!error || typeof error !== 'object') {
    return error ?? null;
  }

  return Object.fromEntries(
    Object.entries(error as Record<string, unknown>).map(([key, value]) => [
      key,
      serializeError(value),
    ]),
  );
};

const normalizeLogContext = (context: unknown): unknown => {
  if (context == null) {
    return undefined;
  }
  if (context instanceof Error) {
    return { error: serializeError(context) };
  }
  if (Array.isArray(context)) {
    return context.map((item) => normalizeLogContext(item));
  }
  if (typeof context === 'object') {
    return Object.fromEntries(
      Object.entries(context as Record<string, unknown>).map(([key, value]) => [
        key,
        normalizeLogContext(value),
      ]),
    );
  }
  return context;
};

const mergeContext = (base: unknown, extra: unknown): unknown => {
  if (!base && !extra) {
    return undefined;
  }
  if (!base) {
    return normalizeLogContext(extra);
  }
  if (!extra) {
    return base;
  }
  const normalized = normalizeLogContext(extra);
  if (!normalized) {
    return base;
  }
  return { ...(base as object), ...(normalized as object) };
};

interface ChatLoggerConfig {
  level?: string;
  service?: string;
}

interface VisitData {
  path?: string;
  referrer?: string;
  sessionId?: string;
  locale?: string;
  title?: string;
  userAgent?: string;
  viewport?: string;
  metadata?: Record<string, unknown>;
}

interface VisitSummary {
  totalVisits: number;
  firstVisit: string | null;
  lastVisit: string | null;
  dailyCounts: Array<{ date: string; count: number }>;
}

interface ChatLogEntry {
  id: number;
  askedAt: string;
  question: string;
  answer: string | null;
  conversationId: string | null;
  model: string | null;
  provider: string | null;
  ragEnabled: boolean | null;
  shortAnswerMode: boolean | null;
  metadata: unknown;
}

interface ChatLogFilter {
  limit?: number;
  offset?: number;
  startAt?: string;
  endAt?: string;
  conversationId?: string;
  model?: string;
  provider?: string;
  ragEnabled?: boolean;
  shortAnswerMode?: boolean;
  search?: string;
}

export interface Logger {
  debug: (message: unknown, context?: unknown) => void;
  info: (message: unknown, context?: unknown) => void;
  warn: (message: unknown, context?: unknown) => void;
  error: (message: unknown, context?: unknown) => void;
  log: (data: unknown) => void;
  logChat: (req: unknown, chatData: unknown) => void;
  logVisit: (visitData?: VisitData) => void;
  getVisitSummary: (filters?: { startAt?: string; endAt?: string; path?: string }) => VisitSummary;
  getChatLogs: (filters?: ChatLogFilter) => ChatLogEntry[];
  clearAllLogs: () => void;
  child: (binding?: unknown) => Logger;
}

class ChatLogger implements Logger {
  private logLevel: LogLevel;
  private baseContext: Record<string, unknown>;
  private dbPath: string;
  private dataDir: string;
  private db: Database.Database;
  private insertChatStmt!: Database.Statement;
  private insertEventStmt!: Database.Statement;
  private buffer: Array<{ type: 'chat'; data: any } | { type: 'event'; data: any }> = [];
  private flushInterval: NodeJS.Timeout;

  constructor({ level, service }: ChatLoggerConfig = {}) {
    this.logLevel = normalizeLevel(level ?? process.env.LOG_LEVEL);
    this.baseContext = {
      service: service ?? 'express-gateway',
      hostname: os.hostname(),
      pid: process.pid,
    };

    this.dbPath = resolveDatabasePath();
    this.dataDir = path.dirname(this.dbPath);

    ensureDirectory(this.dataDir);

    this.db = new Database(this.dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');

    this.initialiseSchema();
    this.prepareStatements();

    // Flush logs every 2 seconds
    this.flushInterval = setInterval(() => this.flush(), 2000);

    process.on('exit', () => {
      this.flush();
      try {
        this.db.close();
      } catch (error) {
        this.emit('error', 'Failed to close analytics database on exit', { error });
      }

      if (process.env.NODE_ENV === 'test' && !process.env.CHAT_LOG_DB_PATH) {
        const cleanupTargets = [this.dbPath, `${this.dbPath}-wal`, `${this.dbPath}-shm`];

        for (const file of cleanupTargets) {
          try {
            if (fs.existsSync(file)) {
              fs.rmSync(file, { force: true });
            }
          } catch (error) {
            this.emit('warn', 'Failed to remove test analytics artifact', {
              file,
              error: serializeError(error),
            });
          }
        }
      }
    });
  }

  flush() {
    if (this.buffer.length === 0) return;

    const currentBatch = [...this.buffer];
    this.buffer = [];

    try {
      this.db.transaction(() => {
        for (const item of currentBatch) {
          try {
            if (item.type === 'chat') {
              this.insertChatStmt.run(item.data);
            } else {
              this.insertEventStmt.run(item.data);
            }
          } catch (error) {
            // We use console.error directly here to avoid infinite recursion if we tried to log this error to the DB
            console.error('Failed to insert log item in batch', error);
          }
        }
      })();
    } catch (error) {
      console.error('Failed to commit log batch transaction', error);
    }
  }

  initialiseSchema() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asked_at INTEGER NOT NULL,
        asked_at_iso TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT,
        conversation_id TEXT,
        model TEXT,
        provider TEXT,
        rag_enabled INTEGER,
        short_answer_mode INTEGER,
        metadata TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_chat_logs_asked_at ON chat_logs (asked_at DESC);
      CREATE INDEX IF NOT EXISTS idx_chat_logs_conversation ON chat_logs (conversation_id);

      CREATE TABLE IF NOT EXISTS event_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at INTEGER NOT NULL,
        recorded_at_iso TEXT NOT NULL,
        type TEXT,
        message TEXT,
        payload TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_event_logs_recorded_at ON event_logs (recorded_at DESC);
    `);
  }

  prepareStatements() {
    this.insertChatStmt = this.db.prepare(`
      INSERT INTO chat_logs (
        asked_at,
        asked_at_iso,
        question,
        answer,
        conversation_id,
        model,
        provider,
        rag_enabled,
        short_answer_mode,
        metadata
      ) VALUES (@asked_at, @asked_at_iso, @question, @answer, @conversation_id, @model, @provider, @rag_enabled, @short_answer_mode, @metadata)
    `);

    this.insertEventStmt = this.db.prepare(`
      INSERT INTO event_logs (
        recorded_at,
        recorded_at_iso,
        type,
        message,
        payload
      ) VALUES (@recorded_at, @recorded_at_iso, @type, @message, @payload)
    `);
  }

  setLevel(level: string) {
    this.logLevel = normalizeLevel(level);
  }

  shouldLog(level: string): boolean {
    const targetLevel = normalizeLevel(level);
    return LEVEL_RANK[targetLevel] >= LEVEL_RANK[this.logLevel];
  }

  emit(level: string, message: unknown, context?: unknown) {
    const resolvedLevel = normalizeLevel(level);
    if (!this.shouldLog(resolvedLevel)) {
      return;
    }

    const entry = {
      timestamp: new Date().toISOString(),
      level: resolvedLevel,
      message: typeof message === 'string' ? message : JSON.stringify(message),
      ...this.baseContext,
    };

    const normalizedContext = normalizeLogContext(context);
    if (normalizedContext && typeof normalizedContext === 'object') {
      Object.assign(entry, normalizedContext);
    }

    const writer =
      resolvedLevel === 'error'
        ? console.error
        : resolvedLevel === 'warn'
          ? console.warn
          : console.log;

    writer(JSON.stringify(entry));
  }

  logWithLevel(level: LogLevel, message: unknown, context?: unknown) {
    const normalizedContext = normalizeLogContext(context);
    this.emit(level, message, normalizedContext);

    if (normalizeLevel(level) === 'debug') {
      return;
    }

    this.recordEvent({
      type: normalizeLevel(level),
      message: typeof message === 'string' ? message : JSON.stringify(message),
      metadata: normalizedContext,
    });
  }

  debug(message: unknown, context?: unknown) {
    this.logWithLevel('debug', message, context);
  }

  info(message: unknown, context?: unknown) {
    this.logWithLevel('info', message, context);
  }

  warn(message: unknown, context?: unknown) {
    this.logWithLevel('warn', message, context);
  }

  error(message: unknown, context?: unknown) {
    this.logWithLevel('error', message, context);
  }

  child(binding: unknown = {}): Logger {
    const bound = normalizeLogContext(binding);
    const combine = (context: unknown) => mergeContext(bound, context);

    return {
      debug: (message: unknown, context?: unknown) => this.debug(message, combine(context)),
      info: (message: unknown, context?: unknown) => this.info(message, combine(context)),
      warn: (message: unknown, context?: unknown) => this.warn(message, combine(context)),
      error: (message: unknown, context?: unknown) => this.error(message, combine(context)),
      log: (data: unknown = {}) => {
        const d = data as Record<string, unknown>;
        this.log({
          ...d,
          metadata: combine(d.metadata ?? d.payload ?? {}),
        });
      },
      logChat: (req: unknown, chatData: unknown = {}) => {
        const cd = chatData as Record<string, unknown>;
        this.logChat(req, {
          ...cd,
          metadata: combine(cd.metadata),
        });
      },
      logVisit: (visitData?: VisitData) => this.logVisit(visitData),
      getVisitSummary: (filters?: { startAt?: string; endAt?: string; path?: string }) => this.getVisitSummary(filters),
      getChatLogs: (filters?: ChatLogFilter) => this.getChatLogs(filters),
      clearAllLogs: () => this.clearAllLogs(),
      child: (extra: unknown = {}) => this.child(mergeContext(bound, extra)),
    };
  }

  recordEvent({
    type,
    message,
    metadata,
  }: {
    type: string | null;
    message: string | null;
    metadata?: unknown;
  }) {
    const now = Date.now();
    const record = {
      recorded_at: now,
      recorded_at_iso: new Date(now).toISOString(),
      type: typeof type === 'string' ? type : null,
      message: typeof message === 'string' ? message : null,
      payload: toMetadataString(metadata),
    };

    this.buffer.push({ type: 'event', data: record });
  }

  logChat(_req: unknown, chatData: unknown) {
    const cd = chatData as Record<string, unknown>;
    if (!cd || !cd.question) {
      this.warn('logChat called without a question payload');
      return;
    }

    const askedAt = cd.timestamp ? Date.parse(cd.timestamp as string) : Date.now();
    const askedAtIso = new Date(askedAt).toISOString();

    const record = {
      asked_at: askedAt,
      asked_at_iso: askedAtIso,
      question: cd.question,
      answer: cd.answer ?? null,
      conversation_id: cd.conversationId ?? cd.conversation_id ?? null,
      model: cd.model ?? null,
      provider: cd.provider ?? null,
      rag_enabled: toBooleanFlag(cd.ragEnabled ?? cd.useRAG),
      short_answer_mode: toBooleanFlag(cd.shortAnswerMode),
      metadata: toMetadataString(cd.metadata),
    };

    this.buffer.push({ type: 'chat', data: record });
  }

  log(data: unknown) {
    if (!data || typeof data !== 'object') {
      this.warn('log called without structured payload');
      return;
    }

    const d = data as Record<string, unknown>;

    const rawMetadata =
      d.metadata ??
      d.payload ??
      (() => {
        const clone = { ...d };
        delete clone.type;
        delete clone.message;
        delete clone.metadata;
        delete clone.payload;
        return clone;
      })();

    this.recordEvent({
      type: typeof d.type === 'string' ? d.type : null,
      message: typeof d.message === 'string' ? d.message : null,
      metadata: normalizeLogContext(rawMetadata),
    });
  }

  logVisit(visitData: VisitData = {}) {
    const pathValue = typeof visitData.path === 'string' ? visitData.path.trim() : '';
    const normalizedPath = pathValue.length > 0 ? pathValue : 'unknown';

    const baseMetadata = {
      path: normalizedPath,
      referrer: typeof visitData.referrer === 'string' ? visitData.referrer : null,
      sessionId: typeof visitData.sessionId === 'string' ? visitData.sessionId : null,
      locale: typeof visitData.locale === 'string' ? visitData.locale : null,
      title: typeof visitData.title === 'string' ? visitData.title : null,
      userAgent: typeof visitData.userAgent === 'string' ? visitData.userAgent : null,
      viewport: typeof visitData.viewport === 'string' ? visitData.viewport : null,
    };

    const additionalMetadata =
      visitData.metadata && typeof visitData.metadata === 'object' ? visitData.metadata : null;

    const metadata = compactObject({
      ...(additionalMetadata || {}),
      ...baseMetadata,
    });

    this.log({
      type: 'visit',
      message: normalizedPath,
      metadata,
    });
  }

  getVisitSummary({
    startAt,
    endAt,
    path,
  }: { startAt?: string; endAt?: string; path?: string } = {}): VisitSummary {
    const conditions = ['type = @eventType'];
    const params: Record<string, unknown> = { eventType: 'visit' };

    if (startAt) {
      const parsedStart = Date.parse(startAt);
      if (!Number.isNaN(parsedStart)) {
        conditions.push('recorded_at >= @startAt');
        params.startAt = parsedStart;
      }
    }

    if (endAt) {
      const parsedEnd = Date.parse(endAt);
      if (!Number.isNaN(parsedEnd)) {
        conditions.push('recorded_at <= @endAt');
        params.endAt = parsedEnd;
      }
    }

    if (typeof path === 'string') {
      const trimmedPath = path.trim();
      if (trimmedPath.length > 0) {
        conditions.push('message = @path');
        params.path = trimmedPath;
      }
    }

    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

    try {
      const totalsStmt = this.db.prepare(`
        SELECT
          COUNT(*) AS totalVisits,
          MIN(recorded_at_iso) AS firstVisit,
          MAX(recorded_at_iso) AS lastVisit
        FROM event_logs
        ${whereClause}
      `);

      const totalsRow = (totalsStmt.get(params) as Record<string, unknown>) || {};

      const trendStmt = this.db.prepare(`
        SELECT
          strftime('%Y-%m-%d', datetime(recorded_at / 1000, 'unixepoch')) AS date,
          COUNT(*) AS count
        FROM event_logs
        ${whereClause}
        GROUP BY date
        ORDER BY date ASC
      `);

      const trendRows = (trendStmt.all(params) as Array<{ date: string; count: number }>) || [];

      return {
        totalVisits: Number(totalsRow.totalVisits) || 0,
        firstVisit: (totalsRow.firstVisit as string) ?? null,
        lastVisit: (totalsRow.lastVisit as string) ?? null,
        dailyCounts: trendRows.map((row) => ({
          date: row.date,
          count: Number(row.count) || 0,
        })),
      };
    } catch (error) {
      this.emit('error', 'Failed to summarise visit analytics', {
        error: serializeError(error),
      });
      return {
        totalVisits: 0,
        firstVisit: null,
        lastVisit: null,
        dailyCounts: [],
      };
    }
  }

  getChatLogs({
    limit = 50,
    offset = 0,
    startAt,
    endAt,
    conversationId,
    model,
    provider,
    ragEnabled,
    shortAnswerMode,
    search,
  }: ChatLogFilter = {}): ChatLogEntry[] {
    const conditions: string[] = [];
    const params: Record<string, unknown> = { limit, offset };

    if (startAt) {
      const parsed = Date.parse(startAt);
      if (!Number.isNaN(parsed)) {
        conditions.push('asked_at >= @startAt');
        params.startAt = parsed;
      }
    }

    if (endAt) {
      const parsed = Date.parse(endAt);
      if (!Number.isNaN(parsed)) {
        conditions.push('asked_at <= @endAt');
        params.endAt = parsed;
      }
    }

    if (conversationId) {
      conditions.push('conversation_id = @conversationId');
      params.conversationId = conversationId;
    }

    if (model) {
      conditions.push('model = @model');
      params.model = model;
    }

    if (provider) {
      conditions.push('provider = @provider');
      params.provider = provider;
    }

    const ragFlag = toBooleanFlag(ragEnabled);
    if (ragFlag !== null) {
      conditions.push('rag_enabled = @ragEnabled');
      params.ragEnabled = ragFlag;
    }

    const shortAnswerFlag = toBooleanFlag(shortAnswerMode);
    if (shortAnswerFlag !== null) {
      conditions.push('short_answer_mode = @shortAnswerMode');
      params.shortAnswerMode = shortAnswerFlag;
    }

    if (search && typeof search === 'string') {
      conditions.push('question LIKE @search');
      params.search = `%${search.trim()}%`;
    }

    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

    try {
      const statement = this.db.prepare(`
        SELECT
          id,
          asked_at_iso as askedAt,
          question,
          answer,
          conversation_id as conversationId,
          model,
          provider,
          rag_enabled as ragEnabled,
          short_answer_mode as shortAnswerMode,
          metadata
        FROM chat_logs
        ${whereClause}
        ORDER BY asked_at DESC
        LIMIT @limit OFFSET @offset
      `);

      return (statement.all(params) as any[]).map((row) => {
        let parsedMetadata = null;
        if (row.metadata) {
          try {
            parsedMetadata = JSON.parse(row.metadata as string);
          } catch (error) {
            this.emit('warn', 'Failed to parse metadata for chat log row', {
              rowId: row.id,
              error: serializeError(error),
            });
            parsedMetadata = row.metadata;
          }
        }

        return {
          id: row.id,
          askedAt: row.askedAt,
          question: row.question,
          answer: row.answer,
          conversationId: row.conversationId,
          model: row.model,
          provider: row.provider,
          ragEnabled: row.ragEnabled === null ? null : row.ragEnabled === 1,
          shortAnswerMode: row.shortAnswerMode === null ? null : row.shortAnswerMode === 1,
          metadata: parsedMetadata,
        };
      });
    } catch (error) {
      this.emit('error', 'Failed to read chat logs', {
        error: serializeError(error),
      });
      return [];
    }
  }

  clearAllLogs() {
    try {
      this.db.exec('DELETE FROM chat_logs; DELETE FROM event_logs;');
    } catch (error) {
      this.emit('error', 'Failed to clear analytics logs', {
        error: serializeError(error),
      });
    }
  }
}

const chatLogger = new ChatLogger();

export const getLogger = (scope?: string) => chatLogger.child(scope ? { scope } : {});

export const rootLogger = chatLogger.child({ scope: 'server' });

export default chatLogger;
