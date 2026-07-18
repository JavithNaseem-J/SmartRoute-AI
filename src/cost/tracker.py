import json
import os
import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.utils.logger import logger


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query = Column(String, nullable=False)
    query_hash = Column(String)
    query_length = Column(Integer)
    model_id = Column(String, nullable=False)
    complexity = Column(String)
    strategy = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cost = Column(Float)
    latency = Column(Float)
    success = Column(Boolean, default=True)


class CostTracker:
    """Cost tracker backed by Supabase PostgreSQL.

    DATABASE_URL must be set. Raises RuntimeError on startup if missing.
    """

    def __init__(self):
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set.\n"
                "This app requires a Supabase PostgreSQL database.\n"
                "1. Create a free project at https://supabase.com\n"
                "2. Go to Project Settings → Database → Connection string → URI\n"
                "3. Set DATABASE_URL=postgresql://... in your .env or Render env vars."
            )

        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,  # detect stale connections before using them
            pool_recycle=300,  # recycle connections every 5 min (Supabase timeout)
        )
        self._Session = sessionmaker(bind=self.engine)
        # NOTE: Schema is managed by Alembic migrations, not create_all().
        # Run `alembic upgrade head` before starting the app (render.yaml does this).
        logger.info(f"CostTracker → Supabase: {database_url.split('@')[-1]}")

    @contextmanager
    def _get_session(self):
        session = self._Session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_query(
        self,
        query: str,
        model_id: str,
        complexity: str,
        strategy: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency: float,
        success: bool = True,
    ):
        q_hash = hashlib.sha256(query.encode()).hexdigest()
        log_entry = QueryLog(
            query=query[:200],
            query_hash=q_hash,
            query_length=len(query),
            model_id=model_id,
            complexity=complexity,
            strategy=strategy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
            success=success,
        )
        with self._get_session() as session:
            session.add(log_entry)
            session.commit()
        logger.info(
            f"Logged: {model_id}, cost=${cost:.4f}, tokens={input_tokens + output_tokens}"
        )

    def get_statistics(self, days: int = 1) -> Dict:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._get_session() as session:
            logs = session.query(QueryLog).filter(QueryLog.timestamp >= cutoff).all()

        if not logs:
            return {
                "total_queries": 0,
                "total_cost": 0.0,
                "avg_cost_per_query": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "avg_latency": 0.0,
                "by_model": {},
                "by_complexity": {},
                "by_strategy": {},
            }

        total_queries = len(logs)
        total_cost = sum(entry.cost for entry in logs)
        by_model: Dict = {}
        by_complexity: Dict = {}
        by_strategy: Dict = {}

        for log in logs:
            for key, bucket in [
                (log.model_id, by_model),
                (log.complexity, by_complexity),
                (log.strategy, by_strategy),
            ]:
                if key not in bucket:
                    bucket[key] = {"count": 0, "cost": 0.0}
                bucket[key]["count"] += 1
                bucket[key]["cost"] += log.cost

        return {
            "total_queries": total_queries,
            "total_cost": round(total_cost, 4),
            "avg_cost_per_query": round(total_cost / total_queries, 4),
            "total_input_tokens": sum(entry.input_tokens for entry in logs),
            "total_output_tokens": sum(entry.output_tokens for entry in logs),
            "avg_latency": round(
                sum(entry.latency for entry in logs) / total_queries, 2
            ),
            "by_model": by_model,
            "by_complexity": by_complexity,
            "by_strategy": by_strategy,
        }

    def calculate_savings(
        self, days: int = 1, baseline_cost_per_query: float = 0.15
    ) -> Dict:
        stats = self.get_statistics(days)
        total_queries = stats["total_queries"]
        actual_cost = stats["total_cost"]
        if total_queries == 0:
            return {
                "baseline_cost": 0.0,
                "actual_cost": 0.0,
                "savings": 0.0,
                "percentage": 0.0,
            }
        baseline_cost = total_queries * baseline_cost_per_query
        savings = baseline_cost - actual_cost
        return {
            "baseline_cost": round(baseline_cost, 4),
            "actual_cost": round(actual_cost, 4),
            "savings": round(savings, 4),
            "percentage": round(
                (savings / baseline_cost * 100) if baseline_cost > 0 else 0, 2
            ),
            "queries": total_queries,
        }

    def get_daily_breakdown(self, days: int = 7) -> Dict:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._get_session() as session:
            logs = session.query(QueryLog).filter(QueryLog.timestamp >= cutoff).all()
        daily = {}
        for log in logs:
            date = log.timestamp.date().isoformat()
            if date not in daily:
                daily[date] = {"queries": 0, "cost": 0.0}
            daily[date]["queries"] += 1
            daily[date]["cost"] += log.cost
        return daily

    def export_to_jsonl(self, output_path: Path, days: int = 30):
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._get_session() as session:
            logs = session.query(QueryLog).filter(QueryLog.timestamp >= cutoff).all()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for log in logs:
                f.write(
                    json.dumps(
                        {
                            "timestamp": log.timestamp.isoformat(),
                            "model_id": log.model_id,
                            "complexity": log.complexity,
                            "cost": log.cost,
                            "latency": log.latency,
                            "success": log.success,
                            "query_hash": log.query_hash,
                        }
                    )
                    + "\n"
                )
        logger.info(f"Exported {len(logs)} logs to {output_path}")

    def close(self):
        if self.engine:
            self.engine.dispose()
