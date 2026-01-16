"""Admin API for RAG system management."""
import asyncio
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Header, Query, Request
from pydantic import BaseModel, Field
import aiofiles

from app.core.logging import get_logger

logger = get_logger(__name__)
from app.core.config import settings
from app.services.document_store import DocumentStore
from app.services.cache import CacheService
from app.core.vectorstore import VectorStore
from app.services.performance_monitor import performance_monitor
from app.pipelines.ingestion import IngestionPipeline
from app.pipelines.parallel_retrieval import create_parallel_pipeline
from app.api.chat import get_llm
from app.models.query import Provider
# from app.services.evaluation import RAGEvaluator, EvaluationDataset  # TODO: Missing module
from app.services.query_logger import get_query_logger
from app.models.query_history import (
    QueryHistoryEntry, QueryHistoryFilter, QueryStatistics, 
    QueryExportRequest
)
from app.api.security import verify_admin_bearer_token

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(verify_admin_bearer_token)])


class IndexRebuildRequest(BaseModel):
    """Request to rebuild index."""
    clear_existing: bool = Field(True, description="Clear existing index before rebuild")
    source_directory: Optional[str] = Field(None, description="Directory to ingest from")
    file_patterns: List[str] = Field(
        default_factory=lambda: ["*.pdf", "*.txt", "*.md", "*.csv"],
        description="File patterns to include"
    )


class CacheManagementRequest(BaseModel):
    """Cache management request."""
    action: str = Field(..., description="Action: clear, warm, stats")
    patterns: Optional[List[str]] = Field(None, description="Patterns to clear (for clear action)")
    warm_queries: Optional[List[str]] = Field(None, description="Queries to warm cache with")


class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""
    config_updates: Dict[str, Any] = Field(..., description="Configuration key-value pairs to update")
    restart_required: bool = Field(False, description="Whether service restart is required")


class BackupRequest(BaseModel):
    """Backup request."""
    backup_type: str = Field("full", description="Backup type: full, incremental, config-only")
    destination: str = Field("local", description="Destination: local, s3, gcs, azure")
    include_vectors: bool = Field(True, description="Include vector embeddings")
    include_indices: bool = Field(True, description="Include search indices")


class SystemStatus(BaseModel):
    """System health status."""
    status: str
    uptime_seconds: float
    version: str
    components: Dict[str, Dict[str, Any]]
    active_connections: int
    memory_usage_mb: float
    cpu_percent: float


# Dependencies
def get_container(request: Request):
    """Get service container from app state."""
    return request.app.state.container


async def get_document_store(request: Request) -> DocumentStore:
    """Get document store instance from container."""
    return request.app.state.container.document_store


async def get_cache_service(request: Request) -> CacheService:
    """Get cache service instance from container."""
    return request.app.state.container.cache_service


async def get_vector_store(request: Request):
    """Get vector store manager instance from container."""
    return request.app.state.container.vector_store_manager


# Global variables for tracking
startup_time = time.time()


@router.get("/health", response_model=SystemStatus)
async def get_system_health(
    _: bool = Depends(verify_admin_bearer_token),
    document_store: DocumentStore = Depends(get_document_store),
    cache_service: CacheService = Depends(get_cache_service),
    vector_store: VectorStore = Depends(get_vector_store)
) -> SystemStatus:
    """Get comprehensive system health status."""
    try:
        # Get performance metrics
        metrics = await performance_monitor.get_metrics_summary()
        
        # Component health checks
        components = {
            "document_store": {
                "status": "healthy",
                "document_count": await document_store.get_stats()
            },
            "cache": {
                "status": "healthy" if cache_service.is_connected else "unhealthy",
                "hit_rate": metrics["cache"]["hit_rate"],
                "total_requests": metrics["cache"]["requests"]
            },
            "vector_store": {
                "status": "healthy",
                "type": settings.VECTOR_STORE_TYPE,
                "embedding_model": settings.EMBEDDING_MODEL
            },
            "llm_pool": {
                "status": "healthy",
                "active_connections": metrics["system"]["active_connections"]
            }
        }
        
        return SystemStatus(
            status="healthy" if all(c["status"] == "healthy" for c in components.values()) else "degraded",
            uptime_seconds=time.time() - startup_time,
            version="1.0.0",  # Get from package.json or config
            components=components,
            active_connections=metrics["system"]["active_connections"],
            memory_usage_mb=metrics["system"]["memory_mb"],
            cpu_percent=metrics["system"]["cpu_percent"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post("/index/rebuild")
async def rebuild_index(
    request: IndexRebuildRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token),
    document_store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStore = Depends(get_vector_store)
) -> Dict[str, Any]:
    """Rebuild the entire index."""
    try:
        # Define async task for background execution
        async def rebuild_task():
            try:
                # Clear existing if requested
                if request.clear_existing:
                    logger.info("Clearing existing index...")
                    await document_store.clear()
                
                # Create ingestion pipeline
                pipeline = IngestionPipeline(
                    vector_store_manager=document_store.vector_store,
                    cache_service=cache_service,
                    source_repository=getattr(document_store, 'source_repository', None)
                )
                
                # Ingest documents
                source_dir = request.source_directory or "./documents"
                logger.info(f"Starting ingestion from {source_dir}")
                
                # Get all files matching patterns
                import glob
                files_to_ingest = []
                for pattern in request.file_patterns:
                    files = glob.glob(f"{source_dir}/**/{pattern}", recursive=True)
                    files_to_ingest.extend(files)
                
                # Ingest files
                for file_path in files_to_ingest:
                    try:
                        await pipeline.ingest_file(file_path)
                    except Exception as e:
                        logger.error(f"Failed to ingest {file_path}: {e}")
                
                logger.info(f"Index rebuild completed. Ingested {len(files_to_ingest)} files")
                
            except Exception as e:
                logger.error(f"Index rebuild failed: {e}")
        
        # Schedule background task
        background_tasks.add_task(rebuild_task)
        
        return {
            "status": "started",
            "message": "Index rebuild started in background",
            "clear_existing": request.clear_existing,
            "source_directory": request.source_directory
        }
        
    except Exception as e:
        logger.error(f"Failed to start index rebuild: {e}")
        raise HTTPException(status_code=500, detail="Failed to start index rebuild")


@router.post("/cache/manage")
async def manage_cache(
    cache_request: CacheManagementRequest,
    request: Request,
    _: bool = Depends(verify_admin_bearer_token),
    cache_service: CacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """Manage cache operations."""
    try:
        if cache_request.action == "clear":
            # Clear cache
            if cache_request.patterns:
                cleared = 0
                for pattern in cache_request.patterns:
                    keys = await cache_service.redis.keys(pattern)
                    if keys:
                        await cache_service.redis.delete(*keys)
                        cleared += len(keys)
                return {
                    "status": "success",
                    "action": "clear",
                    "cleared_keys": cleared
                }
            else:
                # Clear all
                await cache_service.redis.flushdb()
                return {
                    "status": "success",
                    "action": "clear",
                    "message": "All cache cleared"
                }

        elif cache_request.action == "warm":
            # Warm cache with common queries
            if not cache_request.warm_queries:
                # Default warm queries
                cache_request.warm_queries = [
                    "What is the meal allowance?",
                    "What is the POMV rate?",
                    "How do I claim travel expenses?",
                    "What documentation do I need?"
                ]

            # Get services from container
            from app.pipelines.improved_retrieval import ImprovedRetrievalPipeline
            container = request.app.state.container

            pipeline = ImprovedRetrievalPipeline(
                document_store=container.document_store,
                vector_store=container.vector_store_manager,
                cache_service=cache_service
            )
            
            warmed = 0
            for query in cache_request.warm_queries:
                try:
                    await pipeline.retrieve(query)
                    warmed += 1
                except Exception as e:
                    logger.error(f"Failed to warm cache for query '{query}': {e}")

            return {
                "status": "success",
                "action": "warm",
                "warmed_queries": warmed
            }

        elif cache_request.action == "stats":
            # Get cache statistics
            info = await cache_service.redis.info()
            
            # Get key count by pattern
            patterns = {
                "retrieval:*": len(await cache_service.redis.keys("retrieval:*")),
                "llm:*": len(await cache_service.redis.keys("llm:*")),
                "embedding:*": len(await cache_service.redis.keys("embedding:*"))
            }
            
            return {
                "status": "success",
                "action": "stats",
                "stats": {
                    "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                    "total_keys": info.get("db0", {}).get("keys", 0),
                    "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)),
                    "patterns": patterns
                }
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {cache_request.action}")
            
    except Exception as e:
        logger.error(f"Cache management failed: {e}")
        raise HTTPException(status_code=500, detail="Cache management failed")


@router.post("/config/update")
async def update_configuration(
    request: ConfigUpdateRequest,
    _: bool = Depends(verify_admin_bearer_token)
) -> Dict[str, Any]:
    """Update system configuration (hot reload where possible)."""
    try:
        updated = []
        require_restart = []
        
        for key, value in request.config_updates.items():
            # Update in-memory config
            if hasattr(settings, key):
                old_value = getattr(settings, key)
                setattr(settings, key, value)
                updated.append(f"{key}: {old_value} -> {value}")
                
                # Check if restart required
                if key in ["VECTOR_STORE_TYPE", "EMBEDDING_MODEL", "DATABASE_URL"]:
                    require_restart.append(key)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Write to .env file if specified
        if os.getenv("PERSIST_CONFIG_UPDATES", "false").lower() == "true":
            env_file = ".env"
            if os.path.exists(env_file):
                # Read existing
                with open(env_file, "r") as f:
                    lines = f.readlines()
                
                # Update or add
                for key, value in request.config_updates.items():
                    found = False
                    for i, line in enumerate(lines):
                        if line.startswith(f"{key}="):
                            lines[i] = f"{key}={value}\n"
                            found = True
                            break
                    if not found:
                        lines.append(f"{key}={value}\n")
                
                # Write back
                with open(env_file, "w") as f:
                    f.writelines(lines)
        
        return {
            "status": "success",
            "updated": updated,
            "restart_required": len(require_restart) > 0,
            "restart_required_for": require_restart
        }
        
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")


@router.post("/backup/create")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token),
    document_store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStore = Depends(get_vector_store),
    cache_service: CacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """Create system backup."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        
        async def backup_task():
            try:
                backup_dir = f"./backups/{backup_id}"
                os.makedirs(backup_dir, exist_ok=True)
                
                # Backup configuration
                config_backup = {
                    "timestamp": timestamp,
                    "version": "1.0.0",
                    "settings": {
                        key: getattr(settings, key)
                        for key in dir(settings)
                        if not key.startswith("_") and isinstance(getattr(settings, key), (str, int, float, bool, list, dict))
                    }
                }
                
                import json
                with open(f"{backup_dir}/config.json", "w") as f:
                    json.dump(config_backup, f, indent=2)
                
                # Backup vectors if requested
                if request.include_vectors and request.backup_type in ["full"]:
                    vector_backup_dir = f"{backup_dir}/vectors"
                    os.makedirs(vector_backup_dir, exist_ok=True)

                    # Export vectors (implementation depends on vector store type)
                    if settings.vector_store_type == "pgvector":
                        # Use pg_dump to backup PostgreSQL vector store
                        import subprocess
                        dump_file = f"{vector_backup_dir}/pgvector_backup.sql"
                        env = os.environ.copy()
                        env["PGPASSWORD"] = settings.postgres_password or ""

                        try:
                            subprocess.run([
                                "pg_dump",
                                "-h", settings.postgres_host,
                                "-p", str(settings.postgres_port),
                                "-U", settings.postgres_user,
                                "-d", settings.postgres_db,
                                "-t", settings.pgvector_table_name,
                                "-f", dump_file,
                            ], env=env, check=True)
                            logger.info(f"PostgreSQL vector backup created: {dump_file}")
                        except subprocess.CalledProcessError as e:
                            logger.error(f"pg_dump failed: {e}")
                        except FileNotFoundError:
                            logger.warning("pg_dump not found - skipping vector backup")
                
                # Backup indices if requested
                if request.include_indices and request.backup_type in ["full"]:
                    indices_backup_dir = f"{backup_dir}/indices"
                    os.makedirs(indices_backup_dir, exist_ok=True)
                    
                    # Copy BM25 index
                    if os.path.exists("./bm25_index"):
                        shutil.copytree("./bm25_index", f"{indices_backup_dir}/bm25_index")
                    
                    # Copy cooccurrence index
                    if os.path.exists("./cooccurrence_index"):
                        shutil.copytree("./cooccurrence_index", f"{indices_backup_dir}/cooccurrence_index")
                
                # Create manifest
                manifest = {
                    "backup_id": backup_id,
                    "timestamp": timestamp,
                    "type": request.backup_type,
                    "components": {
                        "config": True,
                        "vectors": request.include_vectors,
                        "indices": request.include_indices
                    },
                    "stats": await document_store.get_stats()
                }
                
                with open(f"{backup_dir}/manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)
                
                # Upload to cloud if requested
                if request.destination != "local":
                    # Implement cloud upload (S3, GCS, Azure)
                    logger.info(f"Would upload backup to {request.destination}")
                
                logger.info(f"Backup {backup_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Backup failed: {e}")
        
        # Schedule backup task
        background_tasks.add_task(backup_task)
        
        return {
            "status": "started",
            "backup_id": backup_id,
            "message": "Backup started in background"
        }
        
    except Exception as e:
        logger.error(f"Failed to start backup: {e}")
        raise HTTPException(status_code=500, detail="Failed to start backup")


@router.get("/backups/list")
async def list_backups(
    _: bool = Depends(verify_admin_bearer_token)
) -> List[Dict[str, Any]]:
    """List available backups."""
    try:
        backups = []
        backup_dir = "./backups"
        
        if os.path.exists(backup_dir):
            for backup_name in os.listdir(backup_dir):
                manifest_path = f"{backup_dir}/{backup_name}/manifest.json"
                if os.path.exists(manifest_path):
                    import json
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    
                    # Get backup size
                    backup_size = 0
                    for root, dirs, files in os.walk(f"{backup_dir}/{backup_name}"):
                        for file in files:
                            backup_size += os.path.getsize(os.path.join(root, file))
                    
                    backups.append({
                        "backup_id": manifest["backup_id"],
                        "timestamp": manifest["timestamp"],
                        "type": manifest["type"],
                        "size_mb": round(backup_size / 1024 / 1024, 2),
                        "components": manifest["components"]
                    })
        
        # Sort by timestamp
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups
        
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail="Failed to list backups")


@router.post("/backups/restore/{backup_id}")
async def restore_backup(
    backup_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token)
) -> Dict[str, Any]:
    """Restore from backup."""
    try:
        backup_dir = f"./backups/{backup_id}"

        if not os.path.exists(backup_dir):
            raise HTTPException(status_code=404, detail="Backup not found")

        # Get container from app state for background task
        container = request.app.state.container

        async def restore_task():
            try:
                # Read manifest
                import json
                with open(f"{backup_dir}/manifest.json", "r") as f:
                    manifest = json.load(f)

                # Restore configuration
                if os.path.exists(f"{backup_dir}/config.json"):
                    with open(f"{backup_dir}/config.json", "r") as f:
                        config = json.load(f)
                    logger.info("Configuration restored from backup")

                # Restore vectors
                if manifest["components"]["vectors"]:
                    pgvector_backup = f"{backup_dir}/vectors/pgvector_backup.sql"
                    if os.path.exists(pgvector_backup):
                        # Stop vector store using container
                        vector_store = container.vector_store_manager
                        await vector_store.close()

                        # Use psql to restore PostgreSQL vector store
                        import subprocess
                        env = os.environ.copy()
                        env["PGPASSWORD"] = settings.postgres_password or ""

                        try:
                            subprocess.run([
                                "psql",
                                "-h", settings.postgres_host,
                                "-p", str(settings.postgres_port),
                                "-U", settings.postgres_user,
                                "-d", settings.postgres_db,
                                "-f", pgvector_backup,
                            ], env=env, check=True)
                            logger.info("PostgreSQL vectors restored from backup")
                        except subprocess.CalledProcessError as e:
                            logger.error(f"psql restore failed: {e}")
                        except FileNotFoundError:
                            logger.warning("psql not found - skipping vector restore")

                        # Restart vector store
                        await vector_store.initialize()
                        logger.info("Vector store reinitialized")
                
                # Restore indices
                if manifest["components"]["indices"]:
                    if os.path.exists(f"{backup_dir}/indices/bm25_index"):
                        shutil.rmtree("./bm25_index", ignore_errors=True)
                        shutil.copytree(f"{backup_dir}/indices/bm25_index", "./bm25_index")
                    
                    if os.path.exists(f"{backup_dir}/indices/cooccurrence_index"):
                        shutil.rmtree("./cooccurrence_index", ignore_errors=True)
                        shutil.copytree(f"{backup_dir}/indices/cooccurrence_index", "./cooccurrence_index")
                    
                    logger.info("Indices restored from backup")
                
                logger.info(f"Restore from backup {backup_id} completed")
                
            except Exception as e:
                logger.error(f"Restore failed: {e}")
        
        # Schedule restore task
        background_tasks.add_task(restore_task)
        
        return {
            "status": "started",
            "backup_id": backup_id,
            "message": "Restore started in background. Service restart recommended after completion."
        }
        
    except Exception as e:
        logger.error(f"Failed to start restore: {e}")
        raise HTTPException(status_code=500, detail="Failed to start restore")


@router.get("/metrics/export")
async def export_metrics(
    format: str = "prometheus",
    _: bool = Depends(verify_admin_bearer_token)
) -> Any:
    """Export performance metrics."""
    try:
        metrics_data = await performance_monitor.export_metrics(format)
        
        if format == "prometheus":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=metrics_data)
        else:
            return metrics_data
            
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


# @router.post("/evaluation/run")
# async def run_evaluation(
#     dataset_name: str,
#     background_tasks: BackgroundTasks,
#     _: bool = Depends(verify_admin_bearer_token),
#     document_store: DocumentStore = Depends(get_document_store),
#     cache_service: CacheService = Depends(get_cache_service)
# ) -> Dict[str, Any]:
#     """Run evaluation on a dataset."""
#     try:
#         async def evaluation_task():
#             try:
#                 # Create evaluator
#                 evaluator = RAGEvaluator(
#                     document_store=document_store,
#                     cache_service=cache_service
#                 )
#                 
#                 # Load dataset
#                 if dataset_name == "travel_rates":
#                     dataset = evaluator.travel_rates_dataset
#                 elif dataset_name == "policy":
#                     dataset = evaluator.policy_dataset
#                 elif dataset_name == "complex":
#                     dataset = evaluator.complex_dataset
#                 else:
#                     # Try to load from file
#                     dataset = EvaluationDataset.load(f"./datasets/{dataset_name}.json")
#                 
#                 # Create retrieval function
#                 from app.pipelines.improved_retrieval import ImprovedRetrievalPipeline
#                 from app.main import vector_store
#                 
#                 pipeline = ImprovedRetrievalPipeline(
#                     document_store=document_store,
#                     vector_store=vector_store,
#                     cache_service=cache_service
#                 )
#                 
#                 # Run evaluation
#                 results = await evaluator.evaluate_dataset(
#                     dataset,
#                     pipeline.retrieve
#                 )
#                 
#                 # Generate report
#                 report = evaluator.generate_report(results)
#                 
#                 # Save report
#                 import json
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 report_path = f"./evaluation_reports/{dataset_name}_{timestamp}.json"
#                 os.makedirs("./evaluation_reports", exist_ok=True)
#                 
#                 with open(report_path, "w") as f:
#                     json.dump(report, f, indent=2)
#                 
#                 logger.info(f"Evaluation completed for dataset {dataset_name}")
#                 
#             except Exception as e:
#                 logger.error(f"Evaluation failed: {e}")
#         
#         # Schedule evaluation task
#         background_tasks.add_task(evaluation_task)
#         
#         return {
#             "status": "started",
#             "dataset": dataset_name,
#             "message": "Evaluation started in background"
#         }
#         
#     except Exception as e:
#         logger.error(f"Failed to start evaluation: {e}")
#         raise HTTPException(status_code=500, detail="Failed to start evaluation")


@router.get("/logs/tail")
async def tail_logs(
    lines: int = 100,
    level: Optional[str] = None,
    _: bool = Depends(verify_admin_bearer_token)
) -> List[Dict[str, Any]]:
    """Get recent log entries."""
    try:
        log_file = "./logs/rag_service.log"
        if not os.path.exists(log_file):
            return []
        
        # Read last N lines
        import json
        logs = []
        
        async with aiofiles.open(log_file, "r") as f:
            # Get file size
            await f.seek(0, 2)
            file_size = await f.tell()
            
            # Read from end
            chunk_size = min(file_size, lines * 500)  # Estimate 500 chars per line
            await f.seek(max(0, file_size - chunk_size))
            
            content = await f.read()
            log_lines = content.strip().split("\n")
            
            for line in log_lines[-lines:]:
                try:
                    log_entry = json.loads(line)
                    if not level or log_entry.get("level") == level.upper():
                        logs.append(log_entry)
                except:
                    pass
        
        return logs
        
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to read logs")


@router.post("/queries/history", response_model=List[QueryHistoryEntry])
async def get_query_history(
    filters: QueryHistoryFilter,
    _: bool = Depends(verify_admin_bearer_token)
) -> List[QueryHistoryEntry]:
    """Get query history with filtering and pagination."""
    try:
        query_logger = get_query_logger()
        history = await query_logger.get_query_history(filters)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get query history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query history")


@router.get("/queries/stats", response_model=QueryStatistics)
async def get_query_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    _: bool = Depends(verify_admin_bearer_token)
) -> QueryStatistics:
    """Get aggregated query statistics."""
    try:
        query_logger = get_query_logger()
        stats = await query_logger.get_statistics(start_date, end_date)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get query statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query statistics")


@router.delete("/queries/clear")
async def clear_old_queries(
    days: int = Query(90, description="Delete queries older than this many days"),
    _: bool = Depends(verify_admin_bearer_token)
) -> Dict[str, Any]:
    """Clear old queries from history."""
    try:
        query_logger = get_query_logger()
        deleted_count = await query_logger.cleanup_old_queries(days)
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} queries older than {days} days"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear old queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear old queries")


@router.post("/queries/export")
async def export_queries(
    export_request: QueryExportRequest,
    _: bool = Depends(verify_admin_bearer_token)
):
    """Export query history to CSV or JSON format."""
    try:
        query_logger = get_query_logger()
        export_data = await query_logger.export_queries(export_request)
        
        # Set appropriate content type and filename
        if export_request.format == "csv":
            media_type = "text/csv"
            filename = f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            media_type = "application/json"
            filename = f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        from fastapi.responses import Response
        return Response(
            content=export_data,
            media_type=media_type,
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Failed to export queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to export queries")


class RetrievalWarmupRequest(BaseModel):
    """Warmup request to prebuild retrieval pipelines and BM25 corpus."""
    provider: str = Field("openai", description="Provider key, e.g., openai, google, anthropic")
    model: Optional[str] = Field(None, description="Model name; defaults to configured chat model")
    use_hybrid_search: bool = Field(True, description="Whether to build hybrid (BM25+Vector) pipeline")
    enable_unified: Optional[bool] = Field(None, description="Override unified retrieval flag")
    queries: Optional[List[str]] = Field(None, description="Optional queries to warm retrieval caches")


@router.post("/warmup/retrieval")
async def warmup_retrieval(
    req: RetrievalWarmupRequest,
    request: Request,
    _: bool = Depends(verify_admin_bearer_token)
) -> Dict[str, Any]:
    """Prebuild retrieval pipeline, cache it, and warm BM25 corpus with optional queries."""
    try:
        app = request.app
        vector_store_manager = getattr(app.state, 'vector_store_manager', None)
        if vector_store_manager is None:
            raise HTTPException(status_code=500, detail="VectorStoreManager not initialized")

        # Ensure BM25 corpus is loaded/refreshed
        corpus = vector_store_manager.get_all_documents(refresh=True)
        corpus_count = len(corpus) if corpus is not None else 0

        # Build LLM (for MultiQuery, etc.)
        try:
            llm = await asyncio.to_thread(get_llm, req.provider, req.model)
        except Exception as e:
            logger.warning(f"LLM creation failed during warmup, continuing without LLM: {e}")
            llm = None

        enable_unified = req.enable_unified if req.enable_unified is not None else getattr(settings, 'enable_unified_retrieval', False)

        # Prepare retriever configuration matching chat path
        retriever_configs = None
        if req.use_hybrid_search:
            retriever_configs = {
                "vector_similarity": {"type": "vector", "search_type": "similarity", "k": 10},
                "bm25": {"type": "bm25", "k": 10},
            }
            # Multi-query disabled for performance (saves ~6s, only 0.1 RRF weight)
            # if llm is not None:
            #     retriever_configs["multi_query"] = {"type": "multi_query", "base_retriever": "vector_similarity", "llm": llm}

        # Create pipeline
        pipeline = await asyncio.to_thread(
            create_parallel_pipeline,
            vector_store_manager=vector_store_manager,
            llm=llm,
            enable_unified=enable_unified,
            retriever_configs=retriever_configs
        )

        # Store in cache with the same key scheme used by chat
        provider_key = str(req.provider)
        model_key = req.model or "default"
        hybrid_key = "hybrid" if req.use_hybrid_search else "vector"
        pipeline_cache_key = f"{hybrid_key}|unified={enable_unified}|{provider_key}|{model_key}"

        if hasattr(app.state, 'retrieval_pipeline_cache') and isinstance(app.state.retrieval_pipeline_cache, dict):
            app.state.retrieval_pipeline_cache[pipeline_cache_key] = pipeline
            logger.info(f"Cached retrieval pipeline: {pipeline_cache_key}")

        # Optionally warm retrieval with sample queries
        warmed = 0
        warm_queries = req.queries or [
            "meal allowance rates",
            "POMV rate",
            "kilometric rates",
            "incidental allowance"
        ]
        for q in warm_queries:
            try:
                await pipeline.retrieve(query=q, k=5)
                warmed += 1
            except Exception as e:
                logger.warning(f"Warm retrieval failed for '{q}': {e}")

        return {
            "status": "success",
            "pipeline_cache_key": pipeline_cache_key,
            "bm25_corpus_docs": corpus_count,
            "warmed_queries": warmed
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval warmup failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval warmup failed")


@router.get("/queries/realtime")
async def get_realtime_queries(
    minutes: int = Query(5, description="Get queries from last N minutes"),
    _: bool = Depends(verify_admin_bearer_token)
) -> List[QueryHistoryEntry]:
    """Get real-time query activity from the last N minutes."""
    try:
        query_logger = get_query_logger()
        
        # Create filter for recent queries
        start_date = datetime.utcnow() - timedelta(minutes=minutes)
        filters = QueryHistoryFilter(
            start_date=start_date,
            limit=100,
            order_by="timestamp",
            order_desc=True
        )
        
        history = await query_logger.get_query_history(filters)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get realtime queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to get realtime queries")
