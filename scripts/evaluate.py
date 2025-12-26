import sys
from pathlib import Path
from typing import List, Dict
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.inference import InferencePipeline


def run_evaluation():
    """Run comprehensive evaluation"""
    
    print("="*70)
    print("COST-OPTIMIZED RAG EVALUATION")
    print("="*70)
    
    # Test queries with different complexities
    test_queries = [
        # Simple queries
        "What is machine learning?",
        "Define neural network",
        "Who invented Python?",
        "What does API stand for?",
        "When was AI created?",
        
        # Medium queries
        "How does gradient descent work?",
        "Explain the difference between supervised and unsupervised learning",
        "Compare CNNs and RNNs",
        "Why is deep learning popular?",
        "Describe how transformers work",
        
        # Complex queries
        "Analyze the ethical implications of AI in healthcare",
        "Evaluate different approaches to AGI development",
        "Compare various machine learning optimization algorithms",
        "Synthesize research on neural network architectures",
        "Provide a comprehensive analysis of reinforcement learning"
    ]
    
    print(f"\nRunning evaluation on {len(test_queries)} queries...")
    print("-"*70)
    
    # Initialize pipeline
    pipeline = InferencePipeline()
    
    # Test with different strategies
    strategies = ["cost_optimized", "quality_first"]
    results_by_strategy = {}
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*70}")
        
        results = []
        total_cost = 0
        total_latency = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] Processing: {query[:50]}...")
            
            result = pipeline.run(query, strategy=strategy, use_retrieval=False)
            
            if result['success']:
                print(f"  ✓ Model: {result['model_used']}")
                print(f"  ✓ Complexity: {result['complexity']}")
                print(f"  ✓ Cost: ${result['cost']:.4f}")
                print(f"  ✓ Latency: {result['latency']:.2f}s")
                
                total_cost += result['cost']
                total_latency += result['latency']
                
                results.append({
                    'query': query,
                    'model': result['model_used'],
                    'complexity': result['complexity'],
                    'cost': result['cost'],
                    'latency': result['latency'],
                    'confidence': result['confidence']
                })
            else:
                print(f"  ✗ Error: {result['error']}")
        
        # Calculate metrics
        avg_cost = total_cost / len(test_queries)
        avg_latency = total_latency / len(test_queries)
        
        results_by_strategy[strategy] = {
            'results': results,
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'avg_latency': avg_latency,
            'total_queries': len(test_queries)
        }
        
        print(f"\n{'-'*70}")
        print(f"SUMMARY - {strategy}")
        print(f"{'-'*70}")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Avg Cost/Query: ${avg_cost:.4f}")
        print(f"Avg Latency: {avg_latency:.2f}s")
    
    # Compare strategies
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    if "cost_optimized" in results_by_strategy and "quality_first" in results_by_strategy:
        cost_opt = results_by_strategy["cost_optimized"]
        quality_first = results_by_strategy["quality_first"]
        
        savings = quality_first['total_cost'] - cost_opt['total_cost']
        savings_pct = (savings / quality_first['total_cost'] * 100) if quality_first['total_cost'] > 0 else 0
        
        print(f"\nQuality First Strategy:")
        print(f"  Total Cost: ${quality_first['total_cost']:.4f}")
        print(f"  Avg Cost: ${quality_first['avg_cost']:.4f}")
        
        print(f"\nCost Optimized Strategy:")
        print(f"  Total Cost: ${cost_opt['total_cost']:.4f}")
        print(f"  Avg Cost: ${cost_opt['avg_cost']:.4f}")
        
        print(f"\n💰 SAVINGS:")
        print(f"  Amount: ${savings:.4f}")
        print(f"  Percentage: {savings_pct:.1f}%")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_by_strategy, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Model distribution
    print(f"\n{'='*70}")
    print("MODEL USAGE DISTRIBUTION (Cost Optimized)")
    print(f"{'='*70}")
    
    if "cost_optimized" in results_by_strategy:
        model_counts = {}
        for result in results_by_strategy["cost_optimized"]['results']:
            model = result['model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(test_queries)) * 100
            print(f"  {model}: {count} queries ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    print("✓ EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_evaluation()