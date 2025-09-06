#!/usr/bin/env python3
"""
Sample Selection and Model Blinding Script

This script combines two model outputs, randomizes and blinds model order,
parses scores and reasoning, and selects a subset of samples for evaluation.

Steps:
1. Load and combine two model prediction files
2. Parse model outputs to extract scores and reasoning
3. Randomize and blind model assignments (A/B)
4. Generate title and domain descriptions for samples
5. Select subset based on article coverage strategy
6. Save final evaluation dataset
"""

import json
import random
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import sys

# Add utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import load_json_file, save_json_file


def parse_model_output(output_str: str) -> Dict[str, Any]:
    """
    Parse model output JSON string to extract overall score and reasoning.
    
    Args:
        output_str: JSON string from model output
        
    Returns:
        Dict containing parsed overall_score and overall_reasoning
    """
    try:
        output_data = json.loads(output_str)
        return {
            "overall_score": output_data.get("Overall_score", None),
            "overall_reasoning": output_data.get("Overall_reasoning", "")
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing model output: {e}")
        return {
            "overall_score": None,
            "overall_reasoning": ""
        }


def get_factors_description(domain: str, domain_descriptions: Dict[str, str]) -> str:
    """
    Get domain description from pre-loaded descriptions.
    
    Args:
        domain: Domain name
        domain_descriptions: Pre-loaded domain descriptions dictionary
        
    Returns:
        String description for the domain
    """
    return domain_descriptions.get(domain, f"This domain evaluates factors related to {domain}")


def combine_model_outputs(paschalidis_path: str, meta_llama_path: str) -> List[Dict[str, Any]]:
    """
    Combine outputs from both models into a single dataset.
    Skip samples with invalid JSON or missing overall_score/reasoning.
    
    Args:
        paschalidis_path: Path to Paschalidis-NOC-Lab predictions
        meta_llama_path: Path to meta-llama predictions
        
    Returns:
        List of combined sample data
    """
    print("Loading model predictions...")
    paschalidis_data = load_json_file(paschalidis_path)
    meta_llama_data = load_json_file(meta_llama_path)
    
    print(f"Loaded {len(paschalidis_data)} Paschalidis samples and {len(meta_llama_data)} meta-llama samples")
    
    # Load domain descriptions once
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    descriptions_path = os.path.join(base_dir, "factor_descriptions.json")
    domain_descriptions = load_json_file(descriptions_path)
    
    # Create lookup for meta-llama data by sample_id
    meta_llama_lookup = {sample["sample_id"]: sample for sample in meta_llama_data}
    
    combined_data = []
    skipped_count = 0
    
    for pasch_sample in paschalidis_data:
        sample_id = pasch_sample["sample_id"]
        
        # Find corresponding meta-llama sample
        meta_sample = meta_llama_lookup.get(sample_id)
        if not meta_sample:
            print(f"Warning: No matching meta-llama sample for sample_id {sample_id}")
            skipped_count += 1
            continue
            
        # Parse model outputs
        pasch_parsed = parse_model_output(pasch_sample.get("output", "{}"))
        meta_parsed = parse_model_output(meta_sample.get("output", "{}"))
        gt_parsed = parse_model_output(pasch_sample.get("GT", "{}"))
        
        # Skip if any model output is invalid (missing overall_score or overall_reasoning)
        if (pasch_parsed["overall_score"] is None or 
            meta_parsed["overall_score"] is None or 
            gt_parsed["overall_score"] is None or
            not pasch_parsed["overall_reasoning"] or
            not meta_parsed["overall_reasoning"] or
            not gt_parsed["overall_reasoning"]):
            print(f"Skipping sample {sample_id}: Invalid JSON or missing score/reasoning")
            skipped_count += 1
            continue
        
        # Get factors description using pre-loaded descriptions
        factors_description = get_factors_description(pasch_sample.get("domain", "Unknown"), domain_descriptions)
        
        # Combine data
        combined_sample = {
            "sample_id": sample_id,
            "domain": pasch_sample.get("domain"),
            "article_id": pasch_sample.get("article_id"),
            "factors_description": factors_description,
            "article_content": pasch_sample.get("article_content"),
            "instruction": pasch_sample.get("instruction"),
            "input": pasch_sample.get("input"),
            "reference_overall_score": gt_parsed["overall_score"],
            "reference_overall_reasoning": gt_parsed["overall_reasoning"],
            "paschalidis_output": pasch_sample.get("output", ""),
            "meta_llama_output": meta_sample.get("output", ""),
            "paschalidis_overall_score": pasch_parsed["overall_score"],
            "meta_llama_overall_score": meta_parsed["overall_score"],
            "paschalidis_overall_reasoning": pasch_parsed["overall_reasoning"],
            "meta_llama_overall_reasoning": meta_parsed["overall_reasoning"]
        }
        
        combined_data.append(combined_sample)
    
    print(f"Successfully combined {len(combined_data)} samples")
    print(f"Skipped {skipped_count} samples due to missing data or invalid JSON")
    return combined_data


def randomize_models(combined_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Randomize and blind model assignments for evaluation.
    
    Strategy: Even sample_ids -> Paschalidis=A, Meta=B
             Odd sample_ids -> Paschalidis=B, Meta=A
    
    Args:
        combined_data: List of combined sample data
        
    Returns:
        List of blinded sample data with model_a/model_b assignments
    """
    print("Randomizing and blinding model assignments...")
    
    blinded_data = []
    
    for sample in combined_data:
        sample_id = sample["sample_id"]
        
        # Determine model assignment based on sample_id parity
        if sample_id % 2 == 0:  # Even: Paschalidis=A, Meta=B
            model_a_name = "Paschalidis-NOC-Lab"
            model_b_name = "meta-llama"
            model_a_score = sample["paschalidis_overall_score"]
            model_b_score = sample["meta_llama_overall_score"]
            model_a_reasoning = sample["paschalidis_overall_reasoning"]
            model_b_reasoning = sample["meta_llama_overall_reasoning"]
        else:  # Odd: Paschalidis=B, Meta=A
            model_a_name = "meta-llama"
            model_b_name = "Paschalidis-NOC-Lab"
            model_a_score = sample["meta_llama_overall_score"]
            model_b_score = sample["paschalidis_overall_score"]
            model_a_reasoning = sample["meta_llama_overall_reasoning"]
            model_b_reasoning = sample["paschalidis_overall_reasoning"]
        
        # Create blinded sample
        blinded_sample = {
            "sample_id": sample_id,
            "title": sample["title"],
            "domain": sample["domain"],
            "factors_description": sample["factors_description"],
            "article_content": sample["article_content"],
            "reference_score": sample["reference_score"],
            "model_a_score": str(model_a_score) if model_a_score is not None else "",
            "model_a_reasoning": model_a_reasoning,
            "model_b_score": str(model_b_score) if model_b_score is not None else "",
            "model_b_reasoning": model_b_reasoning,
            # Keep track of actual model assignments (for later analysis)
            "_model_a_actual": model_a_name,
            "_model_b_actual": model_b_name,
            "_article_id": sample["article_id"]
        }
        
        blinded_data.append(blinded_sample)
    
    print(f"Successfully blinded {len(blinded_data)} samples")
    return blinded_data


def select_evaluation_subset(blinded_data: List[Dict[str, Any]], 
                           target_articles: int = 50, 
                           min_samples_per_article: int = 2) -> List[Dict[str, Any]]:
    """
    Select a subset of samples for evaluation using article-based sampling strategy.
    
    Strategy: Select articles with highest sample counts across domains,
             targeting ~15-20 articles that give 100+ samples total.
    
    Args:
        blinded_data: List of blinded sample data
        target_articles: Target number of articles to select
        min_samples_per_article: Minimum samples per article to consider
        
    Returns:
        List of selected samples for evaluation
    """
    print("Selecting evaluation subset...")
    
    # Group samples by article_id
    article_groups = defaultdict(list)
    for sample in blinded_data:
        article_id = sample["_article_id"]
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
    
    # Sort by sample count (descending) and domain coverage (descending)
    article_stats.sort(key=lambda x: (x["sample_count"], x["domain_count"]), reverse=True)
    
    print(f"Article statistics (top 10):")
    for i, stats in enumerate(article_stats[:10]):
        print(f"  Article {stats['article_id']}: {stats['sample_count']} samples, {stats['domain_count']} domains")
    
    # Select top articles that meet criteria
    selected_articles = []
    total_samples = 0
    domain_coverage = set()
    
    for stats in article_stats:
        if len(selected_articles) >= target_articles:
            break
            
        if stats["sample_count"] >= min_samples_per_article:
            selected_articles.append(stats)
            total_samples += stats["sample_count"]
            domain_coverage.update(stats["domains"])
            
            if total_samples >= 100:  # Stop if we have enough samples
                break
    
    # Collect all samples from selected articles
    selected_samples = []
    for article_stats in selected_articles:
        selected_samples.extend(article_stats["samples"])
    
    print(f"Selected {len(selected_articles)} articles with {len(selected_samples)} total samples")
    print(f"Domain coverage: {len(domain_coverage)} domains - {list(domain_coverage)}")
    
    return selected_samples


def main():
    """Main execution function."""
    # File paths
    base_dir = "/home/jingmei/Pandemic_LLM/Human_disease_evaluation"
    paschalidis_path = os.path.join(base_dir, "Result/Paschalidis-NOC-Lab/Llama-3.1-8B-Full-Severity/Severity_data/predictions.json")
    meta_llama_path = os.path.join(base_dir, "Result/meta-llama/Llama-3.1-8B-Instruct/Severity_data/predictions.json")
    
    # Output paths
    output_dir = os.path.join(base_dir, "Result/evaluation_data")
    os.makedirs(output_dir, exist_ok=True)
    
    combined_output_path = os.path.join(output_dir, "combined_model_outputs.json")
    blinded_output_path = os.path.join(output_dir, "blinded_evaluation_data.json")
    selected_output_path = os.path.join(output_dir, "selected_evaluation_samples.json")
    mapping_output_path = os.path.join(output_dir, "model_mapping.json")
    
    # Step 1: Combine model outputs
    print("=" * 60)
    print("STEP 1: Combining model outputs")
    print("=" * 60)
    combined_data = combine_model_outputs(paschalidis_path, meta_llama_path)
    save_json_file(combined_output_path, combined_data)
    print(f"Saved combined data to: {combined_output_path}")
    
    # Step 2: Randomize and blind models
    print("\n" + "=" * 60)
    print("STEP 2: Randomizing and blinding model assignments")
    print("=" * 60)
    blinded_data = randomize_models(combined_data)
    save_json_file(blinded_output_path, blinded_data)
    print(f"Saved blinded data to: {blinded_output_path}")
    
    # Step 3: Select evaluation subset
    print("\n" + "=" * 60)
    print("STEP 3: Selecting evaluation subset")
    print("=" * 60)
    selected_data = select_evaluation_subset(blinded_data)
    
    # Remove internal tracking fields from final output
    final_data = []
    model_mapping = {}
    
    for sample in selected_data:
        sample_id = sample["sample_id"]
        
        # Store mapping for later reference
        model_mapping[sample_id] = {
            "model_a": sample["_model_a_actual"],
            "model_b": sample["_model_b_actual"],
            "article_id": sample["_article_id"]
        }
        
        # Create clean sample without internal fields
        clean_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
        final_data.append(clean_sample)
    
    save_json_file(selected_output_path, final_data)
    save_json_file(mapping_output_path, model_mapping)
    
    print(f"Saved final evaluation data to: {selected_output_path}")
    print(f"Saved model mapping to: {mapping_output_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {len(combined_data)}")
    print(f"Final evaluation samples: {len(final_data)}")
    
    domain_counts = Counter(sample["domain"] for sample in final_data)
    print(f"Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} samples")
        
    print(f"\nFiles created:")
    print(f"  - {combined_output_path}")
    print(f"  - {blinded_output_path}")
    print(f"  - {selected_output_path}")
    print(f"  - {mapping_output_path}")


if __name__ == "__main__":
    main()