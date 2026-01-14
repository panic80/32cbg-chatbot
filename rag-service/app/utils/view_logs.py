#!/usr/bin/env python3
"""
View encrypted query logs with automatic decryption.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.query_logger import QueryLogger
from app.models.query_history import QueryHistoryFilter, QueryStatus


async def view_logs(
    limit: int = 20,
    days_back: int = 7,
    provider: str = None,
    status: str = None,
    search: str = None,
    show_response: bool = False
):
    """View query logs with various filters."""
    
    query_logger = QueryLogger()
    await query_logger.initialize()
    
    # Build filters
    filters = QueryHistoryFilter(
        limit=limit,
        start_date=datetime.utcnow() - timedelta(days=days_back) if days_back else None,
        provider=provider,
        status=QueryStatus(status) if status else None,
        search_query=search,
        order_by="timestamp",
        order_desc=True
    )
    
    # Get query history
    entries = await query_logger.get_query_history(filters)
    
    if not entries:
        print("No queries found matching the criteria.")
        return
    
    # Prepare data for display
    table_data = []
    for entry in entries:
        row = [
            entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            entry.user_query[:50] + "..." if len(entry.user_query) > 50 else entry.user_query,
            entry.provider,
            entry.model,
            "Yes" if entry.use_rag else "No",
            f"{entry.processing_time:.2f}s",
            entry.status.value
        ]
        
        if show_response:
            response_preview = entry.response_preview[:100] + "..." if entry.response_preview and len(entry.response_preview) > 100 else entry.response_preview
            row.append(response_preview or "N/A")
        
        table_data.append(row)
    
    # Display table
    headers = ["Timestamp", "Query", "Provider", "Model", "RAG", "Time", "Status"]
    if show_response:
        headers.append("Response Preview")
    
    print(f"\n=== Query Logs (Last {days_back} days, showing {len(entries)}/{filters.limit} max) ===\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Show statistics
    print(f"\nTotal queries shown: {len(entries)}")
    
    # Get overall statistics
    stats = await query_logger.get_statistics(
        start_date=datetime.utcnow() - timedelta(days=days_back) if days_back else None
    )
    
    print(f"\n=== Statistics (Last {days_back} days) ===")
    print(f"Total queries: {stats.total_queries}")
    print(f"Successful: {stats.successful_queries}")
    print(f"Failed: {stats.failed_queries}")
    print(f"Average processing time: {stats.average_processing_time:.2f}s")
    print(f"Total tokens used: {stats.total_tokens_used:,}")
    
    if stats.queries_per_provider:
        print("\nQueries by provider:")
        for provider, count in stats.queries_per_provider.items():
            print(f"  - {provider}: {count}")


async def view_single_query(query_id: str):
    """View details of a single query."""
    
    query_logger = QueryLogger()
    await query_logger.initialize()
    
    # Find the specific query
    import aiosqlite
    async with aiosqlite.connect(query_logger.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM query_history WHERE id = ?",
            [query_id]
        )
        row = await cursor.fetchone()
        
        if not row:
            print(f"Query with ID {query_id} not found.")
            return
        
        # Decrypt if needed
        user_query = row['user_query']
        response_preview = row['response_preview']
        
        if query_logger.encrypt_queries and query_logger.encryption_service:
            if row['user_query_encrypted'] and user_query.startswith('[ENCRYPTED-'):
                try:
                    user_query = query_logger.encryption_service.decrypt_text(
                        row['user_query_encrypted'],
                        row['user_query_encryption_version'] or 'v1'
                    )
                except Exception as e:
                    user_query = f"[DECRYPTION_FAILED: {e}]"
            
            if row['response_encrypted']:
                try:
                    full_response = query_logger.encryption_service.decrypt_text(
                        row['response_encrypted'],
                        row['response_encryption_version'] or 'v1'
                    )
                    response_preview = full_response
                except Exception as e:
                    response_preview = f"[DECRYPTION_FAILED: {e}]"
        
        # Display details
        print(f"\n=== Query Details ===")
        print(f"ID: {row['id']}")
        print(f"Timestamp: {row['timestamp']}")
        print(f"Provider: {row['provider']}")
        print(f"Model: {row['model']}")
        print(f"Use RAG: {'Yes' if row['use_rag'] else 'No'}")
        print(f"Status: {row['status']}")
        print(f"Processing Time: {row['processing_time']:.2f}s")
        print(f"Tokens Used: {row['tokens_used'] or 'N/A'}")
        print(f"Sources Count: {row['sources_count']}")
        print(f"Conversation ID: {row['conversation_id'] or 'N/A'}")
        
        print(f"\n=== User Query ===")
        print(user_query)
        
        if response_preview:
            print(f"\n=== Response ===")
            print(response_preview)
        
        if row['error_message']:
            print(f"\n=== Error ===")
            print(row['error_message'])
        
        if row['metadata']:
            print(f"\n=== Metadata ===")
            import json
            metadata = json.loads(row['metadata'])
            for key, value in metadata.items():
                print(f"  {key}: {value}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="View encrypted query logs")
    parser.add_argument('--limit', '-l', type=int, default=20, help='Number of queries to show')
    parser.add_argument('--days', '-d', type=int, default=7, help='Days to look back')
    parser.add_argument('--provider', '-p', help='Filter by provider (openai, google, anthropic)')
    parser.add_argument('--status', '-s', help='Filter by status (SUCCESS, ERROR)')
    parser.add_argument('--search', help='Search in queries')
    parser.add_argument('--show-response', '-r', action='store_true', help='Show response preview')
    parser.add_argument('--id', help='View details of specific query by ID')
    
    args = parser.parse_args()
    
    if args.id:
        asyncio.run(view_single_query(args.id))
    else:
        asyncio.run(view_logs(
            limit=args.limit,
            days_back=args.days,
            provider=args.provider,
            status=args.status,
            search=args.search,
            show_response=args.show_response
        ))


if __name__ == "__main__":
    main()