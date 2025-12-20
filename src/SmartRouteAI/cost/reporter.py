"""Cost reporting and analytics"""

from datetime import datetime, timedelta
from typing import Dict
import csv
from src.cost.tracker import CostTracker, QueryLog
from src.utils.helpers import format_cost, format_tokens
from src.utils.logging import logger


class CostReporter:
    """Generate cost reports and analytics"""
    
    def __init__(self, tracker: CostTracker = None):
        self.tracker = tracker or CostTracker()
    
    def generate_daily_report(self) -> str:
        """Generate daily cost report"""
        
        stats = self.tracker.get_statistics(days=1)
        savings = self.tracker.calculate_savings(days=1)
        
        report = f"""
                   DAILY COST REPORT                         
         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

7-DAY SUMMARY
• Total Queries: {stats['total_queries']}
• Total Cost: {format_cost(stats['total_cost'])}
• Avg Cost/Query: {format_cost(stats['avg_cost_per_query'])}
• Avg Latency: {stats['avg_latency']:.2f}s

SAVINGS vs ALL-GPT-4 BASELINE
• Baseline Cost: {format_cost(savings['baseline_cost'])}
• Actual Cost: {format_cost(savings['actual_cost'])}
• Savings: {format_cost(savings['savings'])} ({savings['percentage']:.1f}%)

TOKENS
• Input: {format_tokens(stats['total_input_tokens'])}
• Output: {format_tokens(stats['total_output_tokens'])}
• Total: {format_tokens(stats['total_input_tokens'] + stats['total_output_tokens'])}

COST BY MODEL
"""
        
        for model_id, model_stats in stats['by_model'].items():
            report += f"• {model_id}: {format_cost(model_stats['cost'])} "
            report += f"({model_stats['count']} queries, "
            report += f"avg {format_cost(model_stats['avg_cost'])})\n"
        
        report += "\nCOST BY COMPLEXITY\n"
        
        for complexity, comp_stats in stats['by_complexity'].items():
            report += f"• {complexity}: {format_cost(comp_stats['cost'])} "
            report += f"({comp_stats['count']} queries)\n"
        
        report += "\n" + "="*62 + "\n"
        
        return report
    
    def generate_weekly_report(self) -> str:
        """Generate weekly cost report"""
        
        stats = self.tracker.get_statistics(days=7)
        savings = self.tracker.calculate_savings(days=7)
        
        report = f"""
                   WEEKLY COST REPORT                        
         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
7-DAY SUMMARY
• Total Queries: {stats['total_queries']}
• Total Cost: {format_cost(stats['total_cost'])}
• Avg Cost/Query: {format_cost(stats['avg_cost_per_query'])}
• Daily Avg Queries: {stats['total_queries'] // 7}

SAVINGS
• Total Saved: {format_cost(savings['savings'])}
• Savings Rate: {savings['percentage']:.1f}%
• Cost Reduction: {format_cost(savings['baseline_cost'])} → {format_cost(savings['actual_cost'])}

TOP MODELS BY USAGE
"""
        
        sorted_models = sorted(
            stats['by_model'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for model_id, model_stats in sorted_models[:5]:
            percentage = (model_stats['count'] / stats['total_queries'] * 100) if stats['total_queries'] > 0 else 0
            report += f"• {model_id}: {model_stats['count']} queries ({percentage:.1f}%) - {format_cost(model_stats['cost'])}\n"
        
        report += "\n" + "="*62 + "\n"
        
        return report
    
    def get_model_performance_comparison(self, days: int = 7) -> Dict:
        """Compare model performance metrics"""
        
        stats = self.tracker.get_statistics(days)
        comparison = {}
        
        for model_id, model_stats in stats['by_model'].items():
            comparison[model_id] = {
                'total_queries': model_stats['count'],
                'total_cost': model_stats['cost'],
                'avg_cost': model_stats['avg_cost'],
                'percentage_of_queries': (
                    model_stats['count'] / stats['total_queries'] * 100
                ) if stats['total_queries'] > 0 else 0,
                'percentage_of_cost': (
                    model_stats['cost'] / stats['total_cost'] * 100
                ) if stats['total_cost'] > 0 else 0
            }
        
        return comparison
    
    def export_csv_report(self, filepath: str, days: int = 30):
        """Export detailed report to CSV"""
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        logs = self.tracker.session.query(QueryLog).filter(
            QueryLog.timestamp >= cutoff
        ).all()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Model', 'Complexity', 'Strategy',
                'Input Tokens', 'Output Tokens', 'Cost', 'Latency', 'Success'
            ])
            
            for log in logs:
                writer.writerow([
                    log.timestamp,
                    log.model_id,
                    log.complexity,
                    log.strategy,
                    log.input_tokens,
                    log.output_tokens,
                    log.cost,
                    log.latency,
                    log.success
                ])
        
        logger.info(f"Exported {len(logs)} logs to {filepath}")
    
    def print_summary(self, days: int = 1):
        """Print a quick summary"""
        if days == 1:
            print(self.generate_daily_report())
        elif days == 7:
            print(self.generate_weekly_report())
        else:
            stats = self.tracker.get_statistics(days)
            print(f"\n{days}-Day Summary:")
            print(f"Queries: {stats['total_queries']}")
            print(f"Cost: {format_cost(stats['total_cost'])}")
            print(f"Avg: {format_cost(stats['avg_cost_per_query'])}/query")