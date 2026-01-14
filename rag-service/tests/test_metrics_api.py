import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.metrics import router as metrics_router
from app.services.performance_monitor import get_performance_monitor


@pytest.fixture(autouse=True)
def reset_monitor():
    monitor = get_performance_monitor()
    monitor.reset_metrics()
    yield
    monitor.reset_metrics()


def test_metrics_summary_returns_expected_shape():
    monitor = get_performance_monitor()
    monitor.record_latency("total_request_latency_ms", 1234.0)
    monitor.record_latency("search_latency_ms", 345.0)
    monitor.record_value("context_coverage_rate", 1.0)
    monitor.increment_counter("total_requests", 1)
    monitor.increment_counter("successful_requests", 1)

    app = FastAPI()
    app.include_router(metrics_router, prefix="/api/v1")
    async def _fetch():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/api/v1/metrics/summary")

    response = asyncio.run(_fetch())
    assert response.status_code == 200

    payload = response.json()
    assert "latency" in payload
    assert "quality" in payload
    assert payload["latency"]["answerTime"]["mean"] == pytest.approx(1234.0)
    assert payload["quality"]["contextCoverage"]["mean"] == pytest.approx(1.0)
    assert payload["throughput"]["totalRequests"] == 1
