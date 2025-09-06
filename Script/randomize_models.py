#!/usr/bin/env python3
"""
Randomize and Blind Models Script

This script takes combined model outputs and randomizes/blinds the model assignments
for unbiased evaluation.

Strategy: Even sample_ids -> Paschalidis=A, Meta=B
         Odd sample_ids -> Paschalidis=B, Meta=A
"""

import os
from typing import Dict, List, Any
import sys

# Add utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import load_json_file, save_json_file


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
            "domain": sample["domain"],
            "factors_description": sample["factors_description"],
            "article_content": sample["article_content"],
            "reference_score": str(sample["reference_overall_score"]),
            "reference_reasoning": sample["reference_overall_reasoning"],
            "model_a_score": str(model_a_score),
            "model_a_reasoning": model_a_reasoning,
            "model_b_score": str(model_b_score),
            "model_b_reasoning": model_b_reasoning,
            "model_a_actual": model_a_name,
            "model_b_actual": model_b_name,
            "article_id": sample["article_id"]
        }
        
        blinded_data.append(blinded_sample)
    
    print(f"Successfully blinded {len(blinded_data)} samples")
    return blinded_data


def main():
    """Main execution function."""
    # File paths
    base_dir = "./"
    input_path = os.path.join(base_dir, "Result/evaluation_data/combined_model_outputs.json")
    
    # Output paths
    output_dir = os.path.join(base_dir, "Result/evaluation_data")
    blinded_output_path = os.path.join(output_dir, "randomized_evaluation_data.json")
    mapping_output_path = os.path.join(output_dir, "model_mapping.json")
    
    # Load combined data
    print("=" * 60)
    print("RANDOMIZING AND BLINDING MODEL ASSIGNMENTS")
    print("=" * 60)
    combined_data = load_json_file(input_path)
    print(f"Loaded {len(combined_data)} combined samples")
    
    # Randomize and blind models
    blinded_data = randomize_models(combined_data)
    
    # Create model mapping for reference
    model_mapping = {}
    for sample in blinded_data:
        sample_id = sample["sample_id"]
        model_mapping[sample_id] = {
            "model_a": sample["model_a_actual"],
            "model_b": sample["model_b_actual"],
            "article_id": sample["article_id"]
        }
    
    # Save results
    save_json_file(blinded_output_path, blinded_data)
    save_json_file(mapping_output_path, model_mapping)
    
    print(f"Saved blinded data to: {blinded_output_path}")
    print(f"Saved model mapping to: {mapping_output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples blinded: {len(blinded_data)}")
    print(f"Model A assignments: {sum(1 for s in blinded_data if s['model_a_actual'] == 'Paschalidis-NOC-Lab')} Paschalidis, {sum(1 for s in blinded_data if s['model_a_actual'] == 'meta-llama')} Meta")


if __name__ == "__main__":
    main()
