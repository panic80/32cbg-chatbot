"""Performance monitoring for the RAG system."""

import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class PerformanceMetric:
    """Single performance metric with history."""
    
    def __init__(self, name: str, window_size: int = 1000):
        """Initialize performance metric."""
        self.name = name
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.total_count = 0
        self.total_sum = 0.0
        
    def record(self, value: float, timestamp: Optional[datetime] = None):
        """Record a new value."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.total_count += 1
        self.total_sum += value
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this metric."""
        if not self.values:
            return {
                "count": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p75": 0,
                "p95": 0,
                "p99": 0,
                "rate_per_minute": 0,
                "window_size": 0
            }

        sorted_values = sorted(self.values)
        last_index = len(sorted_values) - 1

        def percentile_index(ratio: float) -> int:
            index = int(len(sorted_values) * ratio)
            if index > last_index:
                return last_index
            return index

        p50_idx = percentile_index(0.5)
        p75_idx = percentile_index(0.75)
        p95_idx = percentile_index(0.95)
        p99_idx = percentile_index(0.99)

        if len(self.timestamps) > 1:
            time_span = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            rate_per_minute = (len(self.values) / time_span) * 60 if time_span > 0 else 0
        else:
            rate_per_minute = 0

        return {
            "count": self.total_count,
            "mean": statistics.mean(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "p50": sorted_values[p50_idx],
            "p75": sorted_values[p75_idx],
            "p95": sorted_values[p95_idx],
            "p99": sorted_values[p99_idx],
            "rate_per_minute": rate_per_minute,
            "window_size": len(self.values)
        }

    def get_recent_samples(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent samples with timestamps for sparkline charts."""
        if not self.values:
            return []

        limit = max(0, min(limit, len(self.values)))
        start_index = len(self.values) - limit
        samples: List[Dict[str, Any]] = []
        for idx in range(start_index, len(self.values)):
            samples.append({
                "value": float(self.values[idx]),
                "timestamp": self.timestamps[idx].isoformat()
            })
        return samples


class PerformanceMonitor:
    """Monitor and track performance metrics for the RAG system."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.start_time = datetime.now(timezone.utc)
        
        # Pre-define common metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize common metrics."""
        # Latency metrics
        self.metrics["retriever_latency_ms"] = PerformanceMetric("retriever_latency_ms")
        self.metrics["llm_latency_ms"] = PerformanceMetric("llm_latency_ms")
        self.metrics["total_request_latency_ms"] = PerformanceMetric("total_request_latency_ms")
        self.metrics["first_token_latency_ms"] = PerformanceMetric("first_token_latency_ms")
        self.metrics["search_latency_ms"] = PerformanceMetric("search_latency_ms")
        self.metrics["context_build_latency_ms"] = PerformanceMetric("context_build_latency_ms")
        self.metrics["answer_generation_latency_ms"] = PerformanceMetric("answer_generation_latency_ms")
        self.metrics["ingestion_latency_ms"] = PerformanceMetric("ingestion_latency_ms")
        self.metrics["ingestion_chunks"] = PerformanceMetric("ingestion_chunks")
        self.metrics["ingestion_invalid_chunks"] = PerformanceMetric("ingestion_invalid_chunks")

        # Cache metrics
        self.metrics["cache_hit_rate"] = PerformanceMetric("cache_hit_rate")

        # Token usage metrics
        self.metrics["tokens_per_request"] = PerformanceMetric("tokens_per_request")
        self.tokens_per_provider: Dict[str, PerformanceMetric] = {}

        # Success/failure rates
        self.metrics["query_success_rate"] = PerformanceMetric("query_success_rate")
        self.metrics["context_coverage_rate"] = PerformanceMetric("context_coverage_rate")
        self.metrics["hallucination_rate"] = PerformanceMetric("hallucination_rate")
        
    def record_latency(self, metric_name: str, latency_ms: float):
        """Record a latency measurement."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = PerformanceMetric(metric_name)
            
        self.metrics[metric_name].record(latency_ms)

    def record_value(self, metric_name: str, value: float):
        """Record a non-latency metric value (e.g., ratios)."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = PerformanceMetric(metric_name)
        self.metrics[metric_name].record(value)
        
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter."""
        self.counters[counter_name] += value
        
    def set_gauge(self, gauge_name: str, value: float):
        """Set a gauge value."""
        self.gauges[gauge_name] = value
        
    def record_cache_hit(self, cache_level: str, hit: bool):
        """Record cache hit/miss."""
        self.increment_counter(f"cache_{cache_level}_{'hits' if hit else 'misses'}")
        
        # Update hit rate metric
        total = self.counters.get(f"cache_{cache_level}_hits", 0) + self.counters.get(f"cache_{cache_level}_misses", 0)
        if total > 0:
            hit_rate = self.counters.get(f"cache_{cache_level}_hits", 0) / total
            self.metrics["cache_hit_rate"].record(hit_rate)
            
    def record_token_usage(self, provider: str, tokens: int):
        """Record token usage by provider."""
        self.increment_counter(f"tokens_{provider}", tokens)
        
        # Record in per-request metric
        self.metrics["tokens_per_request"].record(tokens)
        
        # Record per provider
        if provider not in self.tokens_per_provider:
            self.tokens_per_provider[provider] = PerformanceMetric(f"tokens_{provider}")
        self.tokens_per_provider[provider].record(tokens)
        
    def record_retriever_performance(self, retriever_name: str, latency_ms: float, docs_retrieved: int):
        """Record retriever performance."""
        # Record latency
        metric_name = f"retriever_{retriever_name}_latency_ms"
        if metric_name not in self.metrics:
            self.metrics[metric_name] = PerformanceMetric(metric_name)
        self.metrics[metric_name].record(latency_ms)
        
        # Record document count
        self.increment_counter(f"retriever_{retriever_name}_docs", docs_retrieved)

    def record_ingestion_result(
        self,
        *,
        processing_time_ms: float,
        chunks: int,
        invalid_chunks: int,
        status: str
    ) -> None:
        """Record ingestion-specific metrics."""
        self.record_latency("ingestion_latency_ms", processing_time_ms)
        self.metrics["ingestion_chunks"].record(chunks)
        self.metrics["ingestion_invalid_chunks"].record(invalid_chunks)
        self.increment_counter("ingestion_total", 1)
        self.increment_counter(f"ingestion_{status}", 1)

    def adjust_ingestion_in_progress(self, delta: int) -> None:
        """Adjust the gauge tracking active ingestions."""
        current = self.gauges.get("ingestion_in_progress", 0)
        new_value = max(0, current + delta)
        self.set_gauge("ingestion_in_progress", new_value)
        
    @asynccontextmanager
    async def measure_latency(self, metric_name: str):
        """Context manager to measure latency."""
        start_time = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.record_latency(metric_name, latency_ms)
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        summary = {
            "uptime_seconds": uptime_seconds,
            "start_time": self.start_time.isoformat(),
            "latencies": {},
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "cache_performance": self._get_cache_performance(),
            "retriever_performance": self._get_retriever_performance(),
            "token_usage": self._get_token_usage(),
            "error_rates": self._get_error_rates()
        }

        # Add latency metrics
        for name, metric in self.metrics.items():
            if "latency" in name:
                summary["latencies"][name] = metric.get_stats()

        return summary

    def get_dashboard_metrics(self, recent_points: int = 24) -> Dict[str, Any]:
        """Return structured metrics specifically for the performance dashboard."""
        def with_recent(metric_name: str) -> Dict[str, Any]:
            metric = self.metrics.get(metric_name)
            if not metric:
                return {
                    "count": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p75": 0,
                    "p95": 0,
                    "p99": 0,
                    "rate_per_minute": 0,
                    "window_size": 0,
                    "recent": []
                }

            stats = metric.get_stats()
            stats["recent"] = metric.get_recent_samples(recent_points)
            return stats

        answer_metric = with_recent("total_request_latency_ms")
        now = datetime.now(timezone.utc)

        throughput = {
            "requestsPerMinute": answer_metric.get("rate_per_minute", 0),
            "totalRequests": self.counters.get("total_requests", 0),
            "successfulRequests": self.counters.get("successful_requests", 0),
            "failedRequests": self.counters.get("failed_requests", 0)
        }

        return {
            "latency": {
                "answerTime": answer_metric,
                "searchTime": with_recent("search_latency_ms"),
                "retrievalTime": with_recent("context_build_latency_ms"),
                "answerGeneration": with_recent("answer_generation_latency_ms"),
                "firstToken": with_recent("first_token_latency_ms")
            },
            "quality": {
                "contextCoverage": with_recent("context_coverage_rate"),
                "contextSupport": with_recent("context_support_ratio"),
                "answerToContext": with_recent("answer_to_context_ratio"),
                "hallucinationRate": with_recent("hallucination_rate"),
                "answerTokens": with_recent("answer_token_count"),
                "sourceTokens": with_recent("source_token_count"),
                "sourceCount": with_recent("source_count"),
                "retrievalScores": {
                    "avg": with_recent("retrieval_score_avg"),
                    "max": with_recent("retrieval_score_max"),
                    "min": with_recent("retrieval_score_min"),
                    "std": with_recent("retrieval_score_std"),
                    "gap": with_recent("retrieval_score_gap"),
                },
                "errorRate": self._get_error_rates()
            },
            "throughput": throughput,
            "cache": self._get_cache_performance(),
            "retrievers": self._get_retriever_performance(),
            "tokenUsage": self._get_token_usage(),
            "meta": {
                "windowSize": next(iter(self.metrics.values())).window_size if self.metrics else 0,
                "updatedAt": now.isoformat(),
                "uptimeSeconds": (now - self.start_time).total_seconds()
            }
        }
        
    def _get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        cache_levels = ["l1", "l2", "l3"]
        performance = {}
        
        for level in cache_levels:
            hits = self.counters.get(f"cache_{level}_hits", 0)
            misses = self.counters.get(f"cache_{level}_misses", 0)
            total = hits + misses
            
            performance[level] = {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hits / total if total > 0 else 0
            }
            
        return performance
        
    def _get_retriever_performance(self) -> Dict[str, Any]:
        """Get retriever performance metrics."""
        performance = {}
        
        # Extract unique retriever names
        retriever_names = set()
        for key in self.metrics.keys():
            if key.startswith("retriever_") and "_latency_ms" in key:
                name = key.replace("retriever_", "").replace("_latency_ms", "")
                retriever_names.add(name)
                
        for name in retriever_names:
            metric_key = f"retriever_{name}_latency_ms"
            docs_key = f"retriever_{name}_docs"
            
            if metric_key in self.metrics:
                performance[name] = {
                    "latency": self.metrics[metric_key].get_stats(),
                    "total_docs": self.counters.get(docs_key, 0)
                }
                
        return performance
        
    def _get_token_usage(self) -> Dict[str, Any]:
        """Get token usage metrics."""
        usage = {
            "total_tokens": sum(self.counters.get(f"tokens_{p}", 0) for p in ["openai", "google", "anthropic"]),
            "by_provider": {}
        }
        
        for provider in ["openai", "google", "anthropic"]:
            counter_key = f"tokens_{provider}"
            if counter_key in self.counters:
                usage["by_provider"][provider] = {
                    "total": self.counters[counter_key],
                    "stats": self.tokens_per_provider[provider].get_stats() if provider in self.tokens_per_provider else {}
                }
                
        return usage
        
    def _get_error_rates(self) -> Dict[str, Any]:
        """Get error rate metrics."""
        total_requests = self.counters.get("total_requests", 0)
        failed_requests = self.counters.get("failed_requests", 0)
        
        return {
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "error_rate": failed_requests / total_requests if total_requests > 0 else 0,
            "errors_by_type": {
                "retrieval_errors": self.counters.get("retrieval_errors", 0),
                "llm_errors": self.counters.get("llm_errors", 0),
                "cache_errors": self.counters.get("cache_errors", 0),
                "timeout_errors": self.counters.get("timeout_errors", 0)
            }
        }
        
    async def export_metrics(self, format: str = "prometheus") -> Any:
        """Export metrics in requested format."""
        if format == "json":
            return self.get_metrics_summary()
        elif format == "prometheus":
            return self.export_prometheus_metrics()
        else:
            raise ValueError(f"Unknown format: {format}")

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Add metadata
        lines.append("# HELP rag_uptime_seconds Time since service start")
        lines.append("# TYPE rag_uptime_seconds gauge")
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        lines.append(f"rag_uptime_seconds {uptime}")
        
        # Add counters
        for name, value in self.counters.items():
            safe_name = name.replace("-", "_")
            lines.append(f"# HELP rag_{safe_name}_total Counter for {name}")
            lines.append(f"# TYPE rag_{safe_name}_total counter")
            lines.append(f"rag_{safe_name}_total {value}")
            
        # Add gauges
        for name, value in self.gauges.items():
            safe_name = name.replace("-", "_")
            lines.append(f"# HELP rag_{safe_name} Gauge for {name}")
            lines.append(f"# TYPE rag_{safe_name} gauge")
            lines.append(f"rag_{safe_name} {value}")
            
        # Add latency histograms
        for name, metric in self.metrics.items():
            if "latency" in name:
                stats = metric.get_stats()
                safe_name = name.replace("-", "_")
                
                lines.append(f"# HELP rag_{safe_name} Latency histogram for {name}")
                lines.append(f"# TYPE rag_{safe_name} histogram")
                
                # Add percentiles
                for percentile, value in [("0.5", stats["p50"]), ("0.95", stats["p95"]), ("0.99", stats["p99"])]:
                    lines.append(f'rag_{safe_name}{{quantile="{percentile}"}} {value}')
                    
                lines.append(f"rag_{safe_name}_count {stats['count']}")
                lines.append(f"rag_{safe_name}_sum {stats['count'] * stats['mean']}")
                
        return "\n".join(lines)
        
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self._init_metrics()
        self.start_time = datetime.now(timezone.utc)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor
