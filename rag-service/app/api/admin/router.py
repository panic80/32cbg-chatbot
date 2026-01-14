"""Main admin router combining all sub-routers."""

from fastapi import APIRouter

from app.api.admin.health import router as health_router
from app.api.admin.index import router as index_router
from app.api.admin.cache_admin import router as cache_router
from app.api.admin.backup import router as backup_router
from app.api.admin.metrics import router as metrics_router
from app.api.admin.queries import router as queries_router
from app.api.admin.config import router as config_router
from app.api.admin.bm25 import router as bm25_router

router = APIRouter()

# Include all sub-routers
router.include_router(health_router)
router.include_router(index_router)
router.include_router(cache_router)
router.include_router(backup_router)
router.include_router(metrics_router)
router.include_router(queries_router)
router.include_router(config_router)
router.include_router(bm25_router)
