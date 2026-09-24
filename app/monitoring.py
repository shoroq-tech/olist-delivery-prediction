import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

LOCK = threading.Lock()
STATS = {
    "requests": 0,
    "errors": 0,
    "latencies_ms": deque(maxlen=1000),
    "by_path": {},
    "started": time.time(),
}
mon_logger = logging.getLogger("monitoring")


def setup_monitoring(app):
    @app.middleware("http")
    async def track(request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            ms = (time.perf_counter() - start) * 1000
            with LOCK:
                STATS["requests"] += 1
                if status >= 400:
                    STATS["errors"] += 1
                STATS["latencies_ms"].append(ms)
                path = request.url.path
                STATS["by_path"][path] = STATS["by_path"].get(path, 0) + 1

    @app.get("/metrics")
    def metrics():
        with LOCK:
            lat = sorted(STATS["latencies_ms"])
            total = STATS["requests"]
            errors = STATS["errors"]
            p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
            avg = sum(lat) / len(lat) if lat else 0.0
            return {
                "requests_total": total,
                "errors_total": errors,
                "error_rate": round(errors / total, 4) if total else 0.0,
                "avg_latency_ms": round(avg, 2),
                "p95_latency_ms": round(p95, 2),
                "requests_by_path": dict(STATS["by_path"]),
                "uptime_seconds": round(time.time() - STATS["started"]),
            }


def log_prediction(order, result):
    try:
        path = Path("logs") / "predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "input": order,
            "output": result,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        mon_logger.exception("Could not write prediction log")
