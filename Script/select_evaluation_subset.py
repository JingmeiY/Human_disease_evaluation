#!/usr/bin/env python3
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any
import sys

# Add utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import load_json_file, save_json_file


def select_evaluation_subset(blinded_data: List[Dict[str, Any]], 
                           target_articles: int = 50) -> List[Dict[str, Any]]:
    
    # Group samples by article_id
    article_groups = defaultdict(list)
    for sample in blinded_data:
        article_id = sample["article_id"]
        article_groups[article_id].append(sample)
    
    # Calculate article statistics
    article_stats = []
    for article_id, samples in article_groups.items():
        # Count samples where model_a_score != model_b_score
        disagreement_count = sum(1 for sample in samples 
                               if sample["model_a_score"] != sample["model_b_score"])
        disagreement_percentage = disagreement_count / len(samples) if len(samples) > 0 else 0
        
        article_stats.append({
            "article_id": article_id,
            "sample_count": len(samples),
            "disagreement_percentage": disagreement_percentage,
            "samples": samples
        })
    
    # Sort by sample count (highest to lowest), then by disagreement percentage (highest to lowest)
    article_stats.sort(key=lambda x: (x["sample_count"], x["disagreement_percentage"]), reverse=True)
    
    # Select top N articles and collect their samples
    #TODO: I think this target_articles is not selecting the number of articles, actually it is selecting the number of samples from each article,
    #TODO:But I want to select the number of articles, so I will change the code to select the number of articles.
    selected_samples = []
    for article_stat in article_stats[:target_articles]:
        selected_samples.extend(article_stat["samples"])
    
    return selected_samples


def main():
    base_dir = "./"
    input_path = os.path.join(base_dir, "Result/evaluation_data/randomized_evaluation_data.json")
    
    # Output paths
    output_dir = os.path.join(base_dir, "Result/evaluation_data")
    selected_output_path = os.path.join(output_dir, "selected_evaluation_samples.json")
    
    # Load blinded data
    blinded_data = load_json_file(input_path)
    
    # Select evaluation subset
    selected_data = select_evaluation_subset(blinded_data, target_articles=5)    
    
    # Save results
    save_json_file(selected_output_path, selected_data)
    print(f"Saved final evaluation data to: {selected_output_path}")
    print(f"Final evaluation samples: {len(selected_data)}")
    
    domain_counts = Counter(sample["domain"] for sample in selected_data)
    print(f"Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} samples")


if __name__ == "__main__":
    main()
