"""Persistence layer for canonical source metadata and usage tracking."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple, Dict, Any

import aiosqlite

from app.core.config import settings
from app.core.logging import get_logger
from app.models.source_catalog import SourceCatalogEntry

logger = get_logger(__name__)


class SourceRepository:
    """Stores canonical source metadata and query usage relationships."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.path.join(
            settings.data_directory,
            "source_catalog.db"
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Ensure the underlying SQLite database and tables exist."""
        if self._initialized:
            return

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT,
                    canonical_url TEXT,
                    reference_path TEXT,
                    document_type TEXT,
                    section TEXT,
                    metadata TEXT,
                    last_ingested_at DATETIME
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS source_documents (
                    source_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT,
                    ingested_at DATETIME NOT NULL,
                    PRIMARY KEY (source_id, chunk_id),
                    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS query_sources (
                    query_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    rank INTEGER,
                    snippet TEXT,
                    created_at DATETIME NOT NULL,
                    PRIMARY KEY (query_id, source_id),
                    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_sources_query ON query_sources(query_id)"
            )
            await db.commit()

        self._initialized = True
        logger.info("Source repository initialized at %s", self.db_path)

    async def upsert_entries(self, entries: Iterable[SourceCatalogEntry]) -> None:
        """Upsert a batch of source entries and related chunk mappings."""
        entries_list = list(entries)
        if not entries_list:
            return

        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            for entry in entries_list:
                await self._upsert_single_entry(db, entry)
            await db.commit()

    async def _upsert_single_entry(
        self,
        db: aiosqlite.Connection,
        entry: SourceCatalogEntry
    ) -> None:
        metadata_json = json.dumps(entry.metadata or {})
        await db.execute(
            """
            INSERT INTO sources (
                source_id, title, canonical_url, reference_path, document_type,
                section, metadata, last_ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                title=excluded.title,
                canonical_url=excluded.canonical_url,
                reference_path=excluded.reference_path,
                document_type=excluded.document_type,
                section=excluded.section,
                metadata=excluded.metadata,
                last_ingested_at=excluded.last_ingested_at
            """,
            (
                entry.source_id,
                entry.title,
                entry.canonical_url,
                entry.reference_path,
                entry.document_type,
                entry.section,
                metadata_json,
                entry.last_ingested_at,
            ),
        )

        now = datetime.now(timezone.utc)
        for chunk_id, document_id in entry.chunk_mappings.items():
            await db.execute(
                """
                INSERT INTO source_documents (source_id, chunk_id, document_id, ingested_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, chunk_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    ingested_at=excluded.ingested_at
                """,
                (
                    entry.source_id,
                    chunk_id,
                    document_id,
                    now,
                ),
            )

    async def record_query_sources(
        self,
        query_id: str,
        sources: Iterable[Dict[str, Any]]
    ) -> None:
        """Persist the sources referenced by a specific query response."""
        source_list = list(sources)
        if not source_list:
            return

        await self.initialize()
        timestamp = datetime.now(timezone.utc)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            for index, source_payload in enumerate(source_list):
                source_id = source_payload.get("source_id") or source_payload.get("id")
                if not source_id:
                    continue
                snippet = source_payload.get("text")
                await db.execute(
                    """
                    INSERT INTO query_sources (query_id, source_id, rank, snippet, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(query_id, source_id) DO UPDATE SET
                        rank=excluded.rank,
                        snippet=excluded.snippet,
                        created_at=excluded.created_at
                    """,
                    (query_id, source_id, index, snippet, timestamp),
                )
            await db.commit()

    async def list_sources(
        self,
        *,
        offset: int = 0,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return canonical sources with aggregate counts and total size."""
        await self.initialize()
        filters = filters or {}

        where_clauses = []
        params: List[Any] = []

        if "document_type" in filters and filters["document_type"]:
            where_clauses.append("document_type = ?")
            params.append(filters["document_type"])
        if "search" in filters and filters["search"]:
            where_clauses.append("(title LIKE ? OR canonical_url LIKE ?)")
            search_term = f"%{filters['search']}%"
            params.extend([search_term, search_term])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")

            total = 0
            async with db.execute(
                f"SELECT COUNT(*) FROM sources {where_sql}",
                params,
            ) as cursor:
                row = await cursor.fetchone()
                total = row[0] if row else 0

            query = f"""
                SELECT
                    s.source_id,
                    s.title,
                    s.canonical_url,
                    s.reference_path,
                    s.document_type,
                    s.section,
                    s.metadata,
                    s.last_ingested_at,
                    COUNT(sd.chunk_id) AS chunk_count,
                    COUNT(DISTINCT sd.document_id) AS document_count
                FROM sources s
                LEFT JOIN source_documents sd ON s.source_id = sd.source_id
                {where_sql}
                GROUP BY s.source_id
                ORDER BY s.last_ingested_at DESC
                LIMIT ? OFFSET ?
            """
            async with db.execute(query, params + [limit, offset]) as cursor:
                records = []
                async for row in cursor:
                    metadata = json.loads(row[6]) if row[6] else {}
                    records.append(
                        {
                            "source_id": row[0],
                            "title": row[1],
                            "canonical_url": row[2],
                            "reference_path": row[3],
                            "document_type": row[4],
                            "section": row[5],
                            "metadata": metadata,
                            "last_ingested_at": row[7],
                            "chunk_count": row[8] or 0,
                            "document_count": row[9] or 0,
                        }
                    )

        return records, total

    async def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single source entry including associated chunks."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            async with db.execute(
                """
                SELECT source_id, title, canonical_url, reference_path, document_type,
                       section, metadata, last_ingested_at
                FROM sources WHERE source_id = ?
                """,
                (source_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                metadata = json.loads(row[6]) if row[6] else {}
                source_payload = {
                    "source_id": row[0],
                    "title": row[1],
                    "canonical_url": row[2],
                    "reference_path": row[3],
                    "document_type": row[4],
                    "section": row[5],
                    "metadata": metadata,
                    "last_ingested_at": row[7],
                    "chunks": [],
                }

            async with db.execute(
                """
                SELECT chunk_id, document_id, ingested_at
                FROM source_documents
                WHERE source_id = ?
                ORDER BY ingested_at DESC
                """,
                (source_id,),
            ) as chunk_cursor:
                async for chunk_row in chunk_cursor:
                    source_payload["chunks"].append(
                        {
                            "chunk_id": chunk_row[0],
                            "document_id": chunk_row[1],
                            "ingested_at": chunk_row[2],
                        }
                    )

            return source_payload

    async def get_sources_for_query(self, query_id: str) -> List[Dict[str, Any]]:
        """Return the sources that were referenced in a given query response."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            async with db.execute(
                """
                SELECT qs.source_id, qs.rank, qs.snippet, s.title, s.canonical_url, s.reference_path
                FROM query_sources qs
                LEFT JOIN sources s ON qs.source_id = s.source_id
                WHERE qs.query_id = ?
                ORDER BY qs.rank ASC
                """,
                (query_id,),
            ) as cursor:
                results = []
                async for row in cursor:
                    results.append(
                        {
                            "source_id": row[0],
                            "rank": row[1],
                            "snippet": row[2],
                            "title": row[3],
                            "canonical_url": row[4],
                            "reference_path": row[5],
                        }
                    )
                return results

    async def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics about the source catalog."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")

            stats = {
                "total_sources": 0,
                "total_chunks": 0,
                "total_documents": 0,
                "last_ingested_at": None,
            }

            async with db.execute("SELECT COUNT(*), MAX(last_ingested_at) FROM sources") as cursor:
                row = await cursor.fetchone()
                if row:
                    stats["total_sources"] = row[0] or 0
                    stats["last_ingested_at"] = row[1]

            async with db.execute("SELECT COUNT(*), COUNT(DISTINCT document_id) FROM source_documents") as cursor:
                row = await cursor.fetchone()
                if row:
                    stats["total_chunks"] = row[0] or 0
                    stats["total_documents"] = row[1] or 0

            return stats

    async def clear_all(self) -> None:
        """Delete all sources and related data from the repository."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("DELETE FROM query_sources")
            await db.execute("DELETE FROM source_documents")
            await db.execute("DELETE FROM sources")
            await db.commit()

        logger.info("Source repository cleared")
