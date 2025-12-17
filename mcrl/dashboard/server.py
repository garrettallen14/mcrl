"""
FastAPI server for MCRL training dashboard.

Provides:
- Real-time metrics visualization
- Training status monitoring
- Milestone progress tracking
- Performance profiling

Run: uvicorn mcrl.dashboard.server:app --port 3000
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mcrl.dashboard.metrics import get_collector, MetricsCollector

# Get paths
DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# Create FastAPI app
app = FastAPI(
    title="MCRL Training Dashboard",
    description="Real-time monitoring for MinecraftRL training",
    version="0.1.0",
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """Get current training status."""
    collector = get_collector()
    return {
        "status": "running",
        "current": collector.get_current(),
        "summary": collector.get_summary(),
    }


@app.get("/api/metrics")
async def get_metrics(n: int = 100):
    """Get recent metrics."""
    collector = get_collector()
    return {
        "metrics": collector.get_recent(n),
        "summary": collector.get_summary(),
    }


@app.get("/api/milestones")
async def get_milestones():
    """Get milestone progress."""
    collector = get_collector()
    return {
        "rates": collector.get_milestone_rates(),
        "episode_count": collector._episode_count,
    }


@app.post("/api/metrics")
async def post_metrics(request: Request):
    """Receive metrics from training process."""
    collector = get_collector()
    data = await request.json()
    collector.record(data)
    return {"status": "ok"}


# Store latest analytics for polling
_latest_analytics = {
    "heatmap": [[0] * 16 for _ in range(16)],
    "depth_histogram": [0] * 20,
    "position_stats": {},
    "num_agents": 0,
}


@app.post("/api/analytics")
async def post_analytics(request: Request):
    """Receive agent position analytics from training."""
    global _latest_analytics
    data = await request.json()
    _latest_analytics = data
    # Also notify SSE listeners
    collector = get_collector()
    collector.record({"_analytics": True, **data})
    return {"status": "ok"}


@app.get("/api/analytics")
async def get_analytics():
    """Get latest agent position analytics."""
    return _latest_analytics


@app.get("/api/timeseries/{field}")
async def get_timeseries(field: str, n: int = 1000):
    """Get timeseries data for a metric."""
    collector = get_collector()
    return {
        "field": field,
        "data": collector.get_timeseries(field, n),
    }


@app.get("/api/stream")
async def stream_metrics():
    """Server-Sent Events stream for real-time updates."""
    collector = get_collector()
    
    async def event_generator():
        queue = asyncio.Queue()
        
        def on_update(metrics):
            try:
                queue.put_nowait(metrics)
            except asyncio.QueueFull:
                pass
        
        collector.add_listener(on_update)
        
        try:
            # Send initial state
            yield f"data: {json.dumps(collector.get_current())}\n\n"
            
            while True:
                try:
                    metrics = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(metrics)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            collector.remove_listener(on_update)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for bidirectional communication."""
    await websocket.accept()
    collector = get_collector()
    
    queue = asyncio.Queue(maxsize=100)
    
    def on_update(metrics):
        try:
            queue.put_nowait(metrics)
        except asyncio.QueueFull:
            pass
    
    collector.add_listener(on_update)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "current": collector.get_current(),
            "summary": collector.get_summary(),
        })
        
        while True:
            try:
                metrics = await asyncio.wait_for(queue.get(), timeout=0.5)
                await websocket.send_json({"type": "update", "metrics": metrics})
            except asyncio.TimeoutError:
                pass
                
    except WebSocketDisconnect:
        pass
    finally:
        collector.remove_listener(on_update)


def start_dashboard(port: int = 3000, host: str = "0.0.0.0"):
    """Start the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="MCRL Training Dashboard")
    parser.add_argument("--port", type=int, default=3000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"Starting MCRL Dashboard at http://{args.host}:{args.port}")
    start_dashboard(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
