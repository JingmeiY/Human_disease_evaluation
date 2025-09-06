#!/usr/bin/env python3
"""
Combine Model Outputs Script

This script combines two model prediction files and parses their outputs.

Steps:
1. Load two model prediction files
2. Parse model outputs to extract scores and reasoning
3. Generate domain descriptions for samples
4. Save combined dataset
"""

import json
import os
from typing import Dict, List, Any
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
    output_data = json.loads(output_str)
    return {
        "overall_score": output_data.get("Overall_score", None),
        "overall_reasoning": output_data.get("Overall_reasoning", "")
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


def combine_model_outputs(paschalidis_path: str, meta_llama_path: str, descriptions_path: str) -> List[Dict[str, Any]]:
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
            
        # Parse model outputs - skip if any parsing fails
        try:
            pasch_parsed = parse_model_output(pasch_sample.get("output", "{}"))
            meta_parsed = parse_model_output(meta_sample.get("output", "{}"))
            gt_parsed = parse_model_output(pasch_sample.get("GT", "{}"))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping sample {sample_id}: JSON parsing error - {e}")
            skipped_count += 1
            continue
        
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


def main():
    """Main execution function."""
    # File paths
    base_dir = "./" 
    paschalidis_path = os.path.join(base_dir, "Result/Paschalidis-NOC-Lab/Llama-3.1-8B-Full-Severity/Severity_data/predictions.json")
    meta_llama_path = os.path.join(base_dir, "Result/meta-llama/Llama-3.1-8B-Instruct/Severity_data/predictions.json")
    descriptions_path = os.path.join(base_dir, "factor_descriptions.json")
    
    # Output paths
    output_dir = os.path.join(base_dir, "Result/evaluation_data")
    os.makedirs(output_dir, exist_ok=True)
    
    combined_output_path = os.path.join(output_dir, "combined_model_outputs.json")
    
    # Combine model outputs
    print("=" * 60)
    print("COMBINING MODEL OUTPUTS")
    print("=" * 60)
    combined_data = combine_model_outputs(paschalidis_path, meta_llama_path, descriptions_path)
    save_json_file(combined_output_path, combined_data)
    print(f"Saved combined data to: {combined_output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples combined: {len(combined_data)}")
    
    from collections import Counter
    domain_counts = Counter(sample["domain"] for sample in combined_data)
    print(f"Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} samples")


if __name__ == "__main__":
    main()
