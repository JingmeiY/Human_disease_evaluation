#!/usr/bin/env python3
"""
Select Evaluation Subset Script

This script selects a subset of blinded samples for evaluation using 
article-based sampling strategy.

Strategy: Select articles with highest sample counts across domains,
         targeting ~15-20 articles that give 100+ samples total.
"""

import os
from collections import defaultdict, Counter
from typing import Dict, List, Any
import sys

# Add utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import load_json_file, save_json_file


def select_evaluation_subset(blinded_data: List[Dict[str, Any]], 
                           target_articles: int = 50) -> List[Dict[str, Any]]:
    """
    Select a subset of samples for evaluation using article-based sampling strategy.
    
    Strategy: Select the top articles with the most samples.
    
    Args:
        blinded_data: List of blinded sample data
        target_articles: Number of top articles to select (default: 50)
        
    Returns:
        List of selected samples for evaluation
    """
    print("Selecting evaluation subset...")
    
    # Group samples by article_id
    article_groups = defaultdict(list)
    for sample in blinded_data:
        article_id = sample["article_id"]
        article_groups[article_id].append(sample)
    
    # Calculate article statistics
    article_stats = []
    for article_id, samples in article_groups.items():
        domains = set(sample["domain"] for sample in samples)
        article_stats.append({
            "article_id": article_id,
            "sample_count": len(samples),
            "domain_count": len(domains),
            "domains": list(domains),
            "samples": samples
        })
    
    # Sort by sample count (highest to lowest)
    article_stats.sort(key=lambda x: x["sample_count"], reverse=True)
    
    print(f"Article statistics (top 10):")
    for i, stats in enumerate(article_stats[:10]):
        print(f"  Article {stats['article_id']}: {stats['sample_count']} samples, {stats['domain_count']} domains")
    
    # Select top N articles
    selected_articles = article_stats[:target_articles]
    
    # Collect all samples from selected articles
    selected_samples = []
    total_samples = 0
    domain_coverage = set()
    
    for article_stats in selected_articles:
        selected_samples.extend(article_stats["samples"])
        total_samples += article_stats["sample_count"]
        domain_coverage.update(article_stats["domains"])
    
    print(f"Selected {len(selected_articles)} articles with {total_samples} total samples")
    print(f"Domain coverage: {len(domain_coverage)} domains - {sorted(list(domain_coverage))}")
    
    return selected_samples


def main():
    """Main execution function."""
    # File paths
    base_dir = "./"
    input_path = os.path.join(base_dir, "Result/evaluation_data/randomized_evaluation_data.json")
    
    # Output paths
    output_dir = os.path.join(base_dir, "Result/evaluation_data")
    selected_output_path = os.path.join(output_dir, "selected_evaluation_samples.json")
    
    # Load blinded data
    print("=" * 60)
    print("SELECTING EVALUATION SUBSET")
    print("=" * 60)
    blinded_data = load_json_file(input_path)
    print(f"Loaded {len(blinded_data)} blinded samples")
    
    # Select evaluation subset
    selected_data = select_evaluation_subset(blinded_data, target_articles=500)
    
    # Remove internal tracking fields from final output
    final_data = []
    for sample in selected_data:
        # Create clean sample without internal fields
        clean_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
        final_data.append(clean_sample)
    
    # Save results
    save_json_file(selected_output_path, final_data)
    print(f"Saved final evaluation data to: {selected_output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final evaluation samples: {len(final_data)}")
    
    domain_counts = Counter(sample["domain"] for sample in final_data)
    print(f"Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} samples")


if __name__ == "__main__":
    main()
