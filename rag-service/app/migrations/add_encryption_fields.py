"""
Database migration to add encryption fields to query_history table.
"""

import asyncio
import aiosqlite
import os
import sys
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.encryption import get_encryption_service
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_column_exists(db, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cursor = await db.execute(
        f"PRAGMA table_info({table_name})"
    )
    columns = await cursor.fetchall()
    return any(col[1] == column_name for col in columns)


async def add_encryption_fields():
    """Add encryption fields to the query_history table."""
    db_path = os.path.join(os.path.dirname(__file__), '../../query_history.db')
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return
    
    async with aiosqlite.connect(db_path) as db:
        # Check if encryption fields already exist
        has_encrypted_query = await check_column_exists(db, 'query_history', 'user_query_encrypted')
        has_encrypted_response = await check_column_exists(db, 'query_history', 'response_encrypted')
        
        # Add user_query_encrypted field
        if not has_encrypted_query:
            logger.info("Adding user_query_encrypted field...")
            await db.execute("""
                ALTER TABLE query_history 
                ADD COLUMN user_query_encrypted TEXT
            """)
            await db.execute("""
                ALTER TABLE query_history 
                ADD COLUMN user_query_encryption_version TEXT DEFAULT 'v1'
            """)
        
        # Add response_encrypted field (full response, not just preview)
        if not has_encrypted_response:
            logger.info("Adding response_encrypted field...")
            await db.execute("""
                ALTER TABLE query_history 
                ADD COLUMN response_encrypted TEXT
            """)
            await db.execute("""
                ALTER TABLE query_history 
                ADD COLUMN response_encryption_version TEXT DEFAULT 'v1'
            """)
        
        # Add encryption metadata field
        has_encryption_metadata = await check_column_exists(db, 'query_history', 'encryption_metadata')
        if not has_encryption_metadata:
            logger.info("Adding encryption_metadata field...")
            await db.execute("""
                ALTER TABLE query_history 
                ADD COLUMN encryption_metadata TEXT
            """)
        
        # Add index for encrypted queries (using hash for searching)
        logger.info("Adding index for query hash...")
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_hash 
            ON query_history(user_query_hash)
        """)
        
        await db.commit()
        logger.info("Migration completed successfully!")


async def migrate_existing_data(batch_size: int = 100):
    """
    Migrate existing unencrypted data to encrypted format.
    This is optional and should be run separately.
    """
    db_path = os.path.join(os.path.dirname(__file__), '../../query_history.db')
    encryption_service = get_encryption_service()
    
    async with aiosqlite.connect(db_path) as db:
        # Count total records to migrate
        cursor = await db.execute("""
            SELECT COUNT(*) FROM query_history 
            WHERE user_query_encrypted IS NULL
            AND user_query NOT LIKE '[ENCRYPTED-%]'
            AND user_query NOT LIKE '[ANONYMIZED-%]'
        """)
        total = (await cursor.fetchone())[0]
        
        if total == 0:
            logger.info("No records to migrate")
            return
        
        logger.info(f"Found {total} records to encrypt")
        
        processed = 0
        while processed < total:
            # Get batch of records
            cursor = await db.execute("""
                SELECT id, user_query, response_preview 
                FROM query_history 
                WHERE user_query_encrypted IS NULL
                AND user_query NOT LIKE '[ENCRYPTED-%]'
                AND user_query NOT LIKE '[ANONYMIZED-%]'
                LIMIT ?
            """, (batch_size,))
            
            records = await cursor.fetchall()
            if not records:
                break
            
            # Encrypt each record
            for record_id, query, response in records:
                try:
                    # Encrypt query
                    encrypted_query, query_version = encryption_service.encrypt_text(query)
                    
                    # Encrypt response if exists
                    encrypted_response = None
                    response_version = None
                    if response:
                        encrypted_response, response_version = encryption_service.encrypt_text(response)
                    
                    # Generate query hash for searching
                    query_hash = encryption_service.generate_hash(query)
                    
                    # Update record
                    await db.execute("""
                        UPDATE query_history 
                        SET user_query_encrypted = ?,
                            user_query_encryption_version = ?,
                            response_encrypted = ?,
                            response_encryption_version = ?,
                            user_query_hash = ?,
                            encryption_metadata = ?
                        WHERE id = ?
                    """, (
                        encrypted_query,
                        query_version,
                        encrypted_response,
                        response_version,
                        query_hash,
                        f'{{"migrated_at": "{datetime.utcnow().isoformat()}"}}',
                        record_id
                    ))
                    
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to encrypt record {record_id}: {e}")
            
            await db.commit()
            logger.info(f"Migrated {processed}/{total} records")
        
        logger.info("Migration completed!")


async def verify_migration():
    """Verify the migration was successful."""
    db_path = os.path.join(os.path.dirname(__file__), '../../query_history.db')
    
    async with aiosqlite.connect(db_path) as db:
        # Check table structure
        cursor = await db.execute("PRAGMA table_info(query_history)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_columns = [
            'user_query_encrypted',
            'user_query_encryption_version',
            'response_encrypted', 
            'response_encryption_version',
            'encryption_metadata'
        ]
        
        missing = [col for col in required_columns if col not in column_names]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        logger.info("All encryption columns present ✓")
        
        # Check for unencrypted sensitive data
        cursor = await db.execute("""
            SELECT COUNT(*) FROM query_history
            WHERE user_query_encrypted IS NULL
            AND user_query NOT LIKE '[ENCRYPTED-%]'
            AND user_query NOT LIKE '[ANONYMIZED-%]'
        """)
        unencrypted_count = (await cursor.fetchone())[0]
        
        if unencrypted_count > 0:
            logger.warning(f"Found {unencrypted_count} unencrypted records")
        else:
            logger.info("All records encrypted ✓")
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration for encryption")
    parser.add_argument('--migrate-data', action='store_true', 
                        help='Migrate existing data to encrypted format')
    parser.add_argument('--verify', action='store_true',
                        help='Verify migration status')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for data migration')
    
    args = parser.parse_args()
    
    if args.verify:
        asyncio.run(verify_migration())
    elif args.migrate_data:
        asyncio.run(add_encryption_fields())
        asyncio.run(migrate_existing_data(args.batch_size))
    else:
        asyncio.run(add_encryption_fields())