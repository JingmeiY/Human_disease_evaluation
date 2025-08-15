#!/usr/bin/env python3
"""
Create Evaluation Batches Script

This script splits selected evaluation samples into batches based on articles.
Each batch contains samples from 5 articles, maintaining the original order.

Strategy: Group by article_id, then create batches of 5 articles each.
"""

import os
from collections import defaultdict
from typing import Dict, List, Any
import sys

# Add utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import load_json_file, save_json_file


def create_evaluation_batches(selected_data: List[Dict[str, Any]], 
                            articles_per_batch: int = 5) -> List[List[Dict[str, Any]]]:
    """
    Split selected evaluation data into batches based on articles.
    
    Args:
        selected_data: List of selected evaluation samples
        articles_per_batch: Number of articles per batch (default: 5)
        
    Returns:
        List of batches, where each batch is a list of samples
    """
    print("Creating evaluation batches...")
    
    # Group samples by article_id while maintaining order
    article_groups = defaultdict(list)
    article_order = []  # Track the order of articles as they appear
    
    for sample in selected_data:
        article_id = sample["article_id"]
        if article_id not in article_groups:
            article_order.append(article_id)
        article_groups[article_id].append(sample)
    
    print(f"Found {len(article_order)} unique articles")
    print(f"Total samples: {len(selected_data)}")
    
    # Create batches of articles
    batches = []
    current_batch = []
    articles_in_current_batch = 0
    
    for article_id in article_order:
        # Add all samples from this article to current batch
        current_batch.extend(article_groups[article_id])
        articles_in_current_batch += 1
        
        # If we've reached the target number of articles per batch, start a new batch
        if articles_in_current_batch >= articles_per_batch:
            batches.append(current_batch)
            current_batch = []
            articles_in_current_batch = 0
    
    # Add any remaining samples as the last batch
    if current_batch:
        batches.append(current_batch)
    
    print(f"Created {len(batches)} batches")
    for i, batch in enumerate(batches):
        # Count unique articles in this batch
        unique_articles = len(set(sample["article_id"] for sample in batch))
        print(f"  Batch {i+1}: {len(batch)} samples from {unique_articles} articles")
    
    return batches


def main():
    """Main execution function."""
    # File paths
    base_dir = "./"
    input_path = os.path.join(base_dir, "Result/evaluation_data/selected_evaluation_samples.json")
    
    # Output directory
    output_dir = os.path.join(base_dir, "Result/evaluation_data/batches")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load selected evaluation data
    print("=" * 60)
    print("CREATING EVALUATION BATCHES")
    print("=" * 60)
    selected_data = load_json_file(input_path)
    print(f"Loaded {len(selected_data)} selected samples")
    
    # Create batches
    batches = create_evaluation_batches(selected_data, articles_per_batch=2)
    
    # Save each batch to a separate file
    batch_files = []
    for i, batch in enumerate(batches):
        batch_id = i + 1
        batch_filename = f"evaluation_samples_batch_{batch_id}.json"
        batch_path = os.path.join(output_dir, batch_filename)
        
        save_json_file(batch_path, batch)
        batch_files.append(batch_path)
        print(f"Saved batch {batch_id} to: {batch_path}")
    
    # Create a summary file
    summary = {
        "total_batches": len(batches),
        "articles_per_batch": 5,
        "total_samples": len(selected_data),
        "batch_info": []
    }
    
    for i, batch in enumerate(batches):
        batch_id = i + 1
        unique_articles = list(set(sample["article_id"] for sample in batch))
        unique_domains = list(set(sample["domain"] for sample in batch))
        
        summary["batch_info"].append({
            "batch_id": batch_id,
            "filename": f"evaluation_samples_batch_{batch_id}.json",
            "sample_count": len(batch),
            "article_count": len(unique_articles),
            "article_ids": unique_articles,
            "domains": unique_domains
        })
    
    summary_path = os.path.join(output_dir, "batch_summary.json")
    save_json_file(summary_path, summary)
    print(f"Saved batch summary to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total batches created: {len(batches)}")
    print(f"Articles per batch: 5 (except possibly the last batch)")
    print(f"Total samples distributed: {sum(len(batch) for batch in batches)}")
    print(f"Batch files saved in: {output_dir}")
    
    print(f"\nBatch details:")
    for i, batch in enumerate(batches):
        batch_id = i + 1
        unique_articles = len(set(sample["article_id"] for sample in batch))
        unique_domains = len(set(sample["domain"] for sample in batch))
        print(f"  Batch {batch_id}: {len(batch)} samples, {unique_articles} articles, {unique_domains} domains")


if __name__ == "__main__":
    main()
