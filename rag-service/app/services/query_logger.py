"""Query logging service for persistent storage and analysis of system usage."""

import os
import sqlite3
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import aiosqlite

from app.core.logging import get_logger
from app.core.config import settings
from app.models.query_history import (
    QueryHistoryEntry, QueryHistoryFilter, QueryStatistics, 
    QueryStatus, QueryExportRequest
)
from app.services.encryption import get_encryption_service

logger = get_logger(__name__)


class QueryLogger:
    """Service for logging and analyzing query history."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize query logger."""
        self.db_path = db_path or os.path.join(
            settings.chroma_persist_directory, 
            "query_history.db"
        )
        self.enabled = getattr(settings, 'enable_query_logging', True)
        self.retention_days = getattr(settings, 'query_retention_days', 90)
        self.anonymize_queries = getattr(settings, 'anonymize_query_logs', False)
        self.encrypt_queries = getattr(settings, 'encrypt_query_logs', True)
        self._initialized = False
        self._encryption_service = None
        
    @property
    def encryption_service(self):
        """Lazy load encryption service."""
        if self._encryption_service is None and self.encrypt_queries:
            self._encryption_service = get_encryption_service()
        return self._encryption_service
        
    async def initialize(self):
        """Initialize database and create tables."""
        if self._initialized:
            return
            
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        user_query TEXT NOT NULL,
                        user_query_hash TEXT,
                        user_query_encrypted TEXT,
                        user_query_encryption_version TEXT DEFAULT 'v1',
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        use_rag BOOLEAN NOT NULL,
                        response_preview TEXT,
                        response_encrypted TEXT,
                        response_encryption_version TEXT DEFAULT 'v1',
                        sources_count INTEGER DEFAULT 0,
                        processing_time REAL NOT NULL,
                        tokens_used INTEGER,
                        conversation_id TEXT,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        metadata TEXT,
                        encryption_metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON query_history(timestamp)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_provider ON query_history(provider)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status ON query_history(status)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversation ON query_history(conversation_id)"
                )
                
                await db.commit()
                
            self._initialized = True
            logger.info(f"Query logger initialized with database at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize query logger: {e}")
            self.enabled = False
            
    async def log_query(
        self,
        query_id: str,
        user_query: str,
        provider: str,
        model: str,
        use_rag: bool,
        response: Optional[str] = None,
        sources_count: int = 0,
        processing_time: float = 0.0,
        tokens_used: Optional[int] = None,
        conversation_id: Optional[str] = None,
        status: QueryStatus = QueryStatus.SUCCESS,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a query to the database."""
        if not self.enabled:
            logger.debug(f"Query logging disabled, skipping query {query_id}")
            return
            
        await self.initialize()
        
        try:
            # Prepare data
            query_hash = None
            query_to_store = user_query
            encrypted_query = None
            query_encryption_version = None
            encrypted_response = None
            response_encryption_version = None
            encryption_metadata = {}
            
            # Handle encryption if enabled
            if self.encrypt_queries and self.encryption_service:
                try:
                    # Always generate hash for searching
                    query_hash = self.encryption_service.generate_hash(user_query)
                    
                    # Encrypt query
                    encrypted_query, query_encryption_version = self.encryption_service.encrypt_text(user_query)
                    query_to_store = f"[ENCRYPTED-{query_encryption_version}]"
                    
                    # Encrypt full response if provided
                    if response:
                        encrypted_response, response_encryption_version = self.encryption_service.encrypt_text(response)
                    
                    encryption_metadata = {
                        "encrypted_at": datetime.utcnow().isoformat(),
                        "key_info": self.encryption_service.get_key_info()
                    }
                except Exception as e:
                    logger.error(f"Encryption failed, falling back to plaintext: {e}")
                    # Fall back to non-encrypted storage
                    self.encrypt_queries = False
            
            # Handle anonymization if no encryption
            if self.anonymize_queries and not self.encrypt_queries:
                # Hash the query for privacy
                query_hash = hashlib.sha256(user_query.encode()).hexdigest()
                query_to_store = f"[ANONYMIZED-{query_hash[:8]}]"
                
            response_preview = None
            if response:
                response_preview = response[:500] + "..." if len(response) > 500 else response
                
            metadata_json = json.dumps(metadata or {})
            encryption_metadata_json = json.dumps(encryption_metadata) if encryption_metadata else None
            
            # Insert into database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO query_history (
                        id, timestamp, user_query, user_query_hash, user_query_encrypted,
                        user_query_encryption_version, provider, model, use_rag, 
                        response_preview, response_encrypted, response_encryption_version,
                        sources_count, processing_time, tokens_used, conversation_id, 
                        status, error_message, metadata, encryption_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id,
                    datetime.utcnow(),
                    query_to_store,
                    query_hash,
                    encrypted_query,
                    query_encryption_version,
                    provider,
                    model,
                    use_rag,
                    response_preview,
                    encrypted_response,
                    response_encryption_version,
                    sources_count,
                    processing_time,
                    tokens_used,
                    conversation_id,
                    status.value,
                    error_message,
                    metadata_json,
                    encryption_metadata_json
                ))
                await db.commit()
                
            logger.debug(f"Logged query {query_id} with status {status}")
            
        except Exception as e:
            logger.error(f"Failed to log query {query_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    async def get_query_history(
        self, 
        filters: QueryHistoryFilter
    ) -> List[QueryHistoryEntry]:
        """Retrieve query history with filters."""
        await self.initialize()
        
        try:
            # Build query
            query = "SELECT * FROM query_history WHERE 1=1"
            params = []
            
            if filters.start_date:
                query += " AND timestamp >= ?"
                params.append(filters.start_date)
                
            if filters.end_date:
                query += " AND timestamp <= ?"
                params.append(filters.end_date)
                
            if filters.provider:
                query += " AND provider = ?"
                params.append(filters.provider)
                
            if filters.model:
                query += " AND model = ?"
                params.append(filters.model)
                
            if filters.status:
                query += " AND status = ?"
                params.append(filters.status.value)
                
            if filters.use_rag is not None:
                query += " AND use_rag = ?"
                params.append(filters.use_rag)
                
            if filters.conversation_id:
                query += " AND conversation_id = ?"
                params.append(filters.conversation_id)
                
            if filters.search_query and not self.anonymize_queries:
                query += " AND user_query LIKE ?"
                params.append(f"%{filters.search_query}%")
                
            # Add ordering
            query += f" ORDER BY {filters.order_by}"
            if filters.order_desc:
                query += " DESC"
                
            # Add pagination
            query += f" LIMIT {filters.limit} OFFSET {filters.offset}"
            
            # Execute query
            entries = []
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    async for row in cursor:
                        metadata = json.loads(row['metadata'] or '{}')
                        
                        # Prepare data for entry
                        user_query = row['user_query']
                        response_preview = row['response_preview']
                        
                        # Decrypt if encryption is enabled and encrypted data exists
                        if self.encrypt_queries and self.encryption_service:
                            # Check if query is encrypted
                            if row['user_query_encrypted'] and user_query.startswith('[ENCRYPTED-'):
                                try:
                                    version = row['user_query_encryption_version'] or 'v1'
                                    user_query = self.encryption_service.decrypt_text(
                                        row['user_query_encrypted'], 
                                        version
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to decrypt query {row['id']}: {e}")
                                    user_query = "[DECRYPTION_FAILED]"
                            
                            # Check if response is encrypted
                            if row['response_encrypted']:
                                try:
                                    version = row['response_encryption_version'] or 'v1'
                                    # For preview, show first 500 chars of decrypted response
                                    full_response = self.encryption_service.decrypt_text(
                                        row['response_encrypted'],
                                        version
                                    )
                                    response_preview = full_response[:500] + "..." if len(full_response) > 500 else full_response
                                except Exception as e:
                                    logger.warning(f"Failed to decrypt response for {row['id']}: {e}")
                                    # Keep the existing preview if decryption fails
                        
                        entry = QueryHistoryEntry(
                            id=row['id'],
                            timestamp=datetime.fromisoformat(row['timestamp']),
                            user_query=user_query,
                            provider=row['provider'],
                            model=row['model'],
                            use_rag=bool(row['use_rag']),
                            response_preview=response_preview,
                            sources_count=row['sources_count'],
                            processing_time=row['processing_time'],
                            tokens_used=row['tokens_used'],
                            conversation_id=row['conversation_id'],
                            status=QueryStatus(row['status']),
                            error_message=row['error_message'],
                            metadata=metadata
                        )
                        entries.append(entry)
                        
            return entries
            
        except Exception as e:
            logger.error(f"Failed to retrieve query history: {e}")
            return []
            
    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> QueryStatistics:
        """Get aggregated query statistics."""
        await self.initialize()
        
        try:
            stats = QueryStatistics(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                average_processing_time=0.0,
                total_tokens_used=0,
                queries_per_provider={},
                queries_per_model={},
                rag_enabled_queries=0,
                average_sources_per_query=0.0,
                queries_by_hour={},
                queries_by_day={},
                top_errors=[],
                query_trends=[]
            )
            
            # Build date filter
            date_filter = "WHERE 1=1"
            params = []
            
            if start_date:
                date_filter += " AND timestamp >= ?"
                params.append(start_date)
                
            if end_date:
                date_filter += " AND timestamp <= ?"
                params.append(end_date)
                
            async with aiosqlite.connect(self.db_path) as db:
                # Total queries
                async with db.execute(
                    f"SELECT COUNT(*) as count FROM query_history {date_filter}",
                    params
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.total_queries = row[0]
                    
                # Successful queries
                async with db.execute(
                    f"SELECT COUNT(*) as count FROM query_history {date_filter} AND status = 'success'",
                    params
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.successful_queries = row[0]
                    
                stats.failed_queries = stats.total_queries - stats.successful_queries
                
                # Average processing time
                async with db.execute(
                    f"SELECT AVG(processing_time) as avg_time FROM query_history {date_filter}",
                    params
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.average_processing_time = row[0] or 0.0
                    
                # Total tokens
                async with db.execute(
                    f"SELECT SUM(tokens_used) as total_tokens FROM query_history {date_filter}",
                    params
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.total_tokens_used = row[0] or 0
                    
                # Queries per provider
                async with db.execute(
                    f"SELECT provider, COUNT(*) as count FROM query_history {date_filter} GROUP BY provider",
                    params
                ) as cursor:
                    async for row in cursor:
                        stats.queries_per_provider[row[0]] = row[1]
                        
                # Queries per model
                async with db.execute(
                    f"SELECT model, COUNT(*) as count FROM query_history {date_filter} GROUP BY model",
                    params
                ) as cursor:
                    async for row in cursor:
                        stats.queries_per_model[row[0]] = row[1]
                        
                # RAG enabled queries
                async with db.execute(
                    f"SELECT COUNT(*) as count FROM query_history {date_filter} AND use_rag = 1",
                    params
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.rag_enabled_queries = row[0]
                    
                # Average sources per query
                async with db.execute(
                    f"SELECT AVG(sources_count) as avg_sources FROM query_history {date_filter} AND use_rag = 1",
                    params
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.average_sources_per_query = row[0] or 0.0
                    
                # Queries by hour
                async with db.execute(
                    f"SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM query_history {date_filter} GROUP BY hour",
                    params
                ) as cursor:
                    async for row in cursor:
                        stats.queries_by_hour[int(row[0])] = row[1]
                        
                # Queries by day of week
                async with db.execute(
                    f"SELECT strftime('%w', timestamp) as day, COUNT(*) as count FROM query_history {date_filter} GROUP BY day",
                    params
                ) as cursor:
                    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                    async for row in cursor:
                        stats.queries_by_day[days[int(row[0])]] = row[1]
                        
                # Top errors
                async with db.execute(
                    f"""SELECT error_message, COUNT(*) as count 
                    FROM query_history 
                    {date_filter} AND status != 'success' AND error_message IS NOT NULL
                    GROUP BY error_message 
                    ORDER BY count DESC 
                    LIMIT 10""",
                    params
                ) as cursor:
                    async for row in cursor:
                        stats.top_errors.append({
                            'error': row[0],
                            'count': row[1]
                        })
                        
                # Query trends (last 7 days)
                trend_date = datetime.utcnow() - timedelta(days=7)
                async with db.execute(
                    """SELECT DATE(timestamp) as date, COUNT(*) as count 
                    FROM query_history 
                    WHERE timestamp >= ?
                    GROUP BY date 
                    ORDER BY date""",
                    [trend_date]
                ) as cursor:
                    async for row in cursor:
                        stats.query_trends.append({
                            'date': row[0],
                            'count': row[1]
                        })
                        
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return QueryStatistics(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                average_processing_time=0.0,
                total_tokens_used=0,
                queries_per_provider={},
                queries_per_model={},
                rag_enabled_queries=0,
                average_sources_per_query=0.0,
                queries_by_hour={},
                queries_by_day={},
                top_errors=[],
                query_trends=[]
            )
            
    async def cleanup_old_queries(self, days: Optional[int] = None):
        """Remove queries older than specified days."""
        await self.initialize()
        
        try:
            retention_days = days or self.retention_days
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async with aiosqlite.connect(self.db_path) as db:
                result = await db.execute(
                    "DELETE FROM query_history WHERE timestamp < ?",
                    [cutoff_date]
                )
                await db.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} queries older than {retention_days} days")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old queries: {e}")
            return 0
            
    async def export_queries(
        self, 
        request: QueryExportRequest
    ) -> str:
        """Export queries to CSV or JSON format."""
        await self.initialize()
        
        try:
            queries = await self.get_query_history(request.filters)
            
            if request.anonymize:
                # Additional anonymization for export
                for query in queries:
                    query.user_query = f"[ANONYMIZED-{hashlib.sha256(query.user_query.encode()).hexdigest()[:8]}]"
                    if query.response_preview and not request.include_responses:
                        query.response_preview = "[RESPONSE REMOVED]"
                        
            if request.format == "json":
                # Export as JSON
                export_data = []
                for query in queries:
                    data = query.dict()
                    if not request.include_responses:
                        data.pop('response_preview', None)
                    if not request.include_metadata:
                        data.pop('metadata', None)
                    export_data.append(data)
                    
                return json.dumps(export_data, indent=2, default=str)
                
            else:  # CSV format
                import csv
                import io
                
                output = io.StringIO()
                
                # Define CSV columns
                columns = [
                    'id', 'timestamp', 'user_query', 'provider', 'model',
                    'use_rag', 'sources_count', 'processing_time', 'tokens_used',
                    'conversation_id', 'status', 'error_message'
                ]
                
                if request.include_responses:
                    columns.append('response_preview')
                    
                if request.include_metadata:
                    columns.append('metadata')
                    
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                
                for query in queries:
                    row = {
                        'id': query.id,
                        'timestamp': query.timestamp.isoformat(),
                        'user_query': query.user_query,
                        'provider': query.provider,
                        'model': query.model,
                        'use_rag': query.use_rag,
                        'sources_count': query.sources_count,
                        'processing_time': query.processing_time,
                        'tokens_used': query.tokens_used or '',
                        'conversation_id': query.conversation_id or '',
                        'status': query.status.value,
                        'error_message': query.error_message or ''
                    }
                    
                    if request.include_responses:
                        row['response_preview'] = query.response_preview or ''
                        
                    if request.include_metadata:
                        row['metadata'] = json.dumps(query.metadata)
                        
                    writer.writerow(row)
                    
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"Failed to export queries: {e}")
            raise


# Global query logger instance
query_logger = QueryLogger()


def get_query_logger() -> QueryLogger:
    """Get the global query logger instance."""
    return query_logger