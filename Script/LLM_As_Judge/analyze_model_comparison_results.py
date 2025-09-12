#!/usr/bin/env python3

import json
import re
import sys
import os
import argparse
from typing import Dict, List, Any
from collections import Counter

# Add parent directory to path for importing utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import load_json_file, save_json_file

def parse_model_comparison(model_comparison: Any) -> Dict[str, str]:
    # If it's already a dictionary, return it
    if isinstance(model_comparison, dict):
        return model_comparison
    
    # If it's a string, try to parse as JSON first
    if isinstance(model_comparison, str):
        # Skip error messages
        if model_comparison.startswith("ERROR:"):
            return {"Q1_score_comparison": "UNKNOWN", "Q2_reasoning_comparison": "UNKNOWN"}
        
        try:
            return json.loads(model_comparison)
        except json.JSONDecodeError:
            # If JSON parsing fails, use regex to extract values
            result = {}
            for field, pattern in [
                ("Q1_score_comparison", r'"Q1_score_comparison"\s*:\s*"([A-E])"'),
                ("Q2_reasoning_comparison", r'"Q2_reasoning_comparison"\s*:\s*"([A-E])"')
            ]:
                match = re.search(pattern, model_comparison, re.IGNORECASE)
                result[field] = match.group(1).upper() if match else "UNKNOWN"
            return result
    
    # If it's neither string nor dict, return unknown
    return {"Q1_score_comparison": "UNKNOWN", "Q2_reasoning_comparison": "UNKNOWN"}

def normalize_score(score: str) -> str:
    score = score.strip().upper()
    
    # Direct matches
    if score in ['A', 'B', 'C', 'D', 'E']:
        return score
    
    # Text-based mapping
    score_lower = score.lower()
    
    # Mapping patterns for both Q1 and Q2
    patterns = {
        'A': ['model a clearly', 'model a much', 'a clearly', 'clearly better a', 'a reasoning much'],
        'B': ['model a slightly', 'model a somewhat', 'a slightly', 'slightly better a', 'a reasoning slightly'],
        'C': ['both equivalent', 'both equal', 'equivalent', 'tie', 'similar', 'comparable'],
        'D': ['model b slightly', 'b slightly', 'slightly better b', 'b reasoning slightly'],
        'E': ['model b clearly', 'model b much', 'b clearly', 'clearly better b', 'b reasoning much']
    }
    
    for grade, keywords in patterns.items():
        if any(word in score_lower for word in keywords):
            return grade
    
    return 'UNKNOWN'

def map_to_actual_model(option: str, model_a: str, model_b: str) -> str:
    """Map option (A-E) to actual model name based on randomized assignment"""
    if option == 'A' or option == 'B':  # Model A wins (clearly or slightly)
        return model_a
    elif option == 'C':  # Both equivalent
        return 'both'
    elif option == 'D' or option == 'E':  # Model B wins (slightly or clearly)
        return model_b
    else:
        return 'unknown'

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM-as-Judge model comparison results")
    parser.add_argument("--input", "-i", 
                       default="./Result/LLM_Judge/gemini-2.5-pro/model_comparison_results.json",
                       help="Input JSON file with model comparison results")
    parser.add_argument("--output_dir", "-d", 
                       default="./Result/LLM_Judge/gemini-2.5-pro",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Load and parse data
    try:
        data = load_json_file(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    parsed_samples = []
    errors = 0
    
    # Parse each sample
    for i, sample in enumerate(data):
        sample_id = sample.get('sample_id', i)
        model_a_actual = sample.get('model_a_actual')
        model_b_actual = sample.get('model_b_actual')
        
        try:
            parsed_comparison = parse_model_comparison(sample.get('model_comparison', ''))
            q1_normalized = normalize_score(parsed_comparison.get('Q1_score_comparison', 'UNKNOWN'))
            q2_normalized = normalize_score(parsed_comparison.get('Q2_reasoning_comparison', 'UNKNOWN'))
            
            # Map to actual models
            q1_winner = map_to_actual_model(q1_normalized, model_a_actual, model_b_actual)
            q2_winner = map_to_actual_model(q2_normalized, model_a_actual, model_b_actual)
            
            parsed_samples.append({
                'sample_id': sample_id,
                'model_a_actual': model_a_actual,       
                'domain': sample.get('domain'),
                'model_b_actual': model_b_actual,
                'q1_normalized': q1_normalized,
                'q2_normalized': q2_normalized,
                'q1_winner': q1_winner,
                'q2_winner': q2_winner,
            })
        except:
            errors += 1
            parsed_samples.append({
                'sample_id': sample_id,
                "domain": sample.get('domain'),
                'model_a_actual': model_a_actual,
                'model_b_actual': model_b_actual,
                'q1_normalized': 'UNKNOWN',
                'q2_normalized': 'UNKNOWN',
                'q1_winner': 'unknown',
                'q2_winner': 'unknown',
            })

    
    # Count results by actual model
    q1_model_counts = Counter(s['q1_winner'] for s in parsed_samples)
    q2_model_counts = Counter(s['q2_winner'] for s in parsed_samples)
    
    total = len(data)
    
    # Print results
    print(f"Total: {total}, Errors: {errors}")

    # Print model-specific results
    for model in ['meta-llama', 'Paschalidis-NOC-Lab', 'both']:
        q1_count = q1_model_counts.get(model, 0)
        print(f"\n\n{model}: Q1={q1_count} ({q1_count/total*100:.2f}%)")

    # Print model-specific results
    for model in ['meta-llama', 'Paschalidis-NOC-Lab', 'both']:
        q2_count = q2_model_counts.get(model, 0)
        print(f"\n\n{model}: Q2={q2_count} ({q2_count/total*100:.2f}%)")


    # Save files
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    save_json_file(os.path.join(args.output_dir, f"{base_name}_parsed_samples.json"), parsed_samples)

    analysis = {
        'total_samples': total,
        'parsing_errors': errors,
        'q1_model_counts': dict(q1_model_counts),
        'q2_model_counts': dict(q2_model_counts)
    }
    
    save_json_file(os.path.join(args.output_dir, f"{base_name}_analysis.json"), analysis)

if __name__ == "__main__":
    main() 
