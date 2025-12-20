import json
from src.pipeline.inference import HybridInferencePipeline
from src.evaluation.evaluator import RAGEvaluator
from src.utils.logging import logger
from src.utils.helpers import save_json, ensure_dir

TEST_QUERIES = [
    "What is machine learning?",
    "How does neural network training work?",
    "Explain supervised learning",
    "What are transformers in NLP?",
    "Compare gradient descent and Adam",
    "What is deep learning?",
    "How do CNNs work?",
    "Explain backpropagation",
    "What is reinforcement learning?",
    "Define overfitting"
]


def main():
    logger.info("Starting evaluation...")
    
    pipeline = HybridInferencePipeline()
    evaluator = RAGEvaluator()
    
    strategies = ["cost_optimized", "quality_first"]
    
    logger.info(f"Comparing {len(strategies)} strategies on {len(TEST_QUERIES)} queries...")
    
    comparison = evaluator.compare_strategies(
        test_queries=TEST_QUERIES,
        strategies=strategies,
        pipeline=pipeline
    )
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for strategy, results in comparison.items():
        print(f"\n{strategy.upper()}")
        print("-"*70)
        print(f"Total Cost: ${results['total_cost']:.4f}")
        print(f"Avg Cost: ${results['avg_cost']:.4f}")
        print(f"Avg Latency: {results['avg_latency']:.2f}s")
        print("\nRAGAS Scores:")
        for metric, score in results['average_scores'].items():
            print(f"  {metric}: {score:.3f}")
    
    if "cost_optimized" in comparison and "quality_first" in comparison:
        cost_opt = comparison["cost_optimized"]
        quality = comparison["quality_first"]
        
        savings = quality['total_cost'] - cost_opt['total_cost']
        savings_pct = (savings / quality['total_cost'] * 100) if quality['total_cost'] > 0 else 0
        
        print("\n" + "="*70)
        print(f"SAVINGS: ${savings:.4f} ({savings_pct:.1f}%)")
        print("="*70)
    
    # Save results
    ensure_dir("experiments/results")
    output_file = "experiments/results/evaluation_results.json"
    save_json(comparison, output_file)
    logger.info(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()