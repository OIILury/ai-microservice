import asyncio
from fastapi import APIRouter
from app.services.metrics_store import metrics_store

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])

@router.get("/recent")
async def get_recent_metrics(limit: int = 200):
    """Retourne les métriques brutes les plus récentes."""
    return await asyncio.to_thread(metrics_store.get_all_metrics, limit=limit)

@router.get("/stats")
async def get_stats():
    """Retourne les statistiques agrégées (par modèle, par type de tâche, évolution temporelle)."""
    return await asyncio.to_thread(metrics_store.get_aggregated_stats)
