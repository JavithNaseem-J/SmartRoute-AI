import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..utils.logger import logger

Base = declarative_base()


class QueryLog(Base):
    """Database model for query logs"""
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    complexity = Column(String)
    strategy = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cost = Column(Float)
    latency = Column(Float)
    success = Column(Boolean, default=True)


class CostTracker:
    """Track and analyze costs per query"""
    
    def __init__(self, db_path: str = "data/costs/usage.db"):
        # Setup database
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"Cost tracker initialized with DB: {self.db_path}")
    
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
        success: bool = True
    ):
        """Log a query with its cost"""
        
        log_entry = QueryLog(
            query=query[:200],  # Truncate long queries
            model_id=model_id,
            complexity=complexity,
            strategy=strategy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
            success=success
        )
        
        self.session.add(log_entry)
        self.session.commit()
        
        logger.info(
            f"Logged query: {model_id}, "
            f"cost=${cost:.4f}, "
            f"tokens={input_tokens + output_tokens}"
        )
    
    def get_statistics(self, days: int = 1) -> Dict:
        """Get cost statistics for last N days"""
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        logs = self.session.query(QueryLog).filter(
            QueryLog.timestamp >= cutoff
        ).all()
        
        if not logs:
            return {
                'total_queries': 0,
                'total_cost': 0.0,
                'avg_cost_per_query': 0.0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'avg_latency': 0.0,
                'by_model': {},
                'by_complexity': {},
                'by_strategy': {}
            }
        
        # Calculate totals
        total_queries = len(logs)
        total_cost = sum(log.cost for log in logs)
        total_input_tokens = sum(log.input_tokens for log in logs)
        total_output_tokens = sum(log.output_tokens for log in logs)
        avg_latency = sum(log.latency for log in logs) / total_queries
        
        # Group by model
        by_model = {}
        for log in logs:
            if log.model_id not in by_model:
                by_model[log.model_id] = {'count': 0, 'cost': 0.0}
            by_model[log.model_id]['count'] += 1
            by_model[log.model_id]['cost'] += log.cost
        
        # Add average cost per model
        for model_id in by_model:
            count = by_model[model_id]['count']
            by_model[model_id]['avg_cost'] = by_model[model_id]['cost'] / count
        
        # Group by complexity
        by_complexity = {}
        for log in logs:
            if log.complexity not in by_complexity:
                by_complexity[log.complexity] = {'count': 0, 'cost': 0.0}
            by_complexity[log.complexity]['count'] += 1
            by_complexity[log.complexity]['cost'] += log.cost
        
        # Group by strategy
        by_strategy = {}
        for log in logs:
            if log.strategy not in by_strategy:
                by_strategy[log.strategy] = {'count': 0, 'cost': 0.0}
            by_strategy[log.strategy]['count'] += 1
            by_strategy[log.strategy]['cost'] += log.cost
        
        return {
            'total_queries': total_queries,
            'total_cost': round(total_cost, 4),
            'avg_cost_per_query': round(total_cost / total_queries, 4),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'avg_latency': round(avg_latency, 2),
            'by_model': by_model,
            'by_complexity': by_complexity,
            'by_strategy': by_strategy
        }
    
    def calculate_savings(
        self,
        days: int = 1,
        baseline_cost_per_query: float = 0.15
    ) -> Dict:
        """
        Calculate savings vs baseline (all-GPT-4)
        
        Args:
            days: Number of days to analyze
            baseline_cost_per_query: Cost if using expensive model for all queries
        
        Returns:
            Savings statistics
        """
        
        stats = self.get_statistics(days)
        total_queries = stats['total_queries']
        actual_cost = stats['total_cost']
        
        if total_queries == 0:
            return {
                'baseline_cost': 0.0,
                'actual_cost': 0.0,
                'savings': 0.0,
                'percentage': 0.0
            }
        
        baseline_cost = total_queries * baseline_cost_per_query
        savings = baseline_cost - actual_cost
        percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return {
            'baseline_cost': round(baseline_cost, 4),
            'actual_cost': round(actual_cost, 4),
            'savings': round(savings, 4),
            'percentage': round(percentage, 2),
            'queries': total_queries
        }
    
    def get_daily_breakdown(self, days: int = 7) -> Dict:
        """Get daily cost breakdown"""
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        logs = self.session.query(QueryLog).filter(
            QueryLog.timestamp >= cutoff
        ).all()
        
        # Group by day
        daily_costs = {}
        for log in logs:
            date = log.timestamp.date().isoformat()
            if date not in daily_costs:
                daily_costs[date] = {'queries': 0, 'cost': 0.0}
            daily_costs[date]['queries'] += 1
            daily_costs[date]['cost'] += log.cost
        
        return daily_costs
    
    def export_to_jsonl(self, output_path: Path, days: int = 30):
        """Export logs to JSONL for analysis"""
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        logs = self.session.query(QueryLog).filter(
            QueryLog.timestamp >= cutoff
        ).all()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for log in logs:
                entry = {
                    'timestamp': log.timestamp.isoformat(),
                    'model_id': log.model_id,
                    'complexity': log.complexity,
                    'strategy': log.strategy,
                    'input_tokens': log.input_tokens,
                    'output_tokens': log.output_tokens,
                    'cost': log.cost,
                    'latency': log.latency,
                    'success': log.success
                }
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Exported {len(logs)} logs to {output_path}")
    
    def close(self):
        """Close database connection"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()