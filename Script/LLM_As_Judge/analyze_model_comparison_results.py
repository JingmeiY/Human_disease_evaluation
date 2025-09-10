#!/usr/bin/env python3
"""
Analyze LLM-as-Judge model comparison results following the workflow:
1. Load the JSON data
2. Parse model_comparison using JSON or regex fallback
3. Normalize options to A, B, C, D, E
4. Map options to actual model names correctly
5. Count and calculate percentages, print concise summary
"""

import json
import re
import sys
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Add parent directory to path for importing utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import load_json_file

def load_comparison_data(file_path: str) -> List[Dict[str, Any]]:
    """
    TODO1: Load the JSON data
    """
    try:
        data = load_json_file(file_path)
        print(f"✓ Loaded {len(data)} samples from {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

def parse_model_comparison(model_comparison: Any) -> Dict[str, str]:
    """
    TODO2: Parse model_comparison using either JSON string, or regex fallback
    """
    # If it's already a dictionary, return it
    if isinstance(model_comparison, dict):
        return model_comparison
    
    # If it's a string, try to parse as JSON first
    if isinstance(model_comparison, str):
        # Skip error messages
        if model_comparison.startswith("ERROR:"):
            return {"Q1_score_comparison": "UNKNOWN", "Q2_reasoning_comparison": "UNKNOWN"}
        
        try:
            # Try to parse as JSON
            parsed = json.loads(model_comparison)
            return parsed
        except json.JSONDecodeError:
            # If JSON parsing fails, use regex to extract values
            result = {}
            
            # Extract Q1_score_comparison
            q1_match = re.search(r'"Q1_score_comparison"\s*:\s*"([A-E])"', model_comparison, re.IGNORECASE)
            if q1_match:
                result["Q1_score_comparison"] = q1_match.group(1).upper()
            else:
                result["Q1_score_comparison"] = "UNKNOWN"
            
            # Extract Q2_reasoning_comparison
            q2_match = re.search(r'"Q2_reasoning_comparison"\s*:\s*"([A-E])"', model_comparison, re.IGNORECASE)
            if q2_match:
                result["Q2_reasoning_comparison"] = q2_match.group(1).upper()
            else:
                result["Q2_reasoning_comparison"] = "UNKNOWN"
            
            return result
    
    # If it's neither string nor dict, return unknown
    return {"Q1_score_comparison": "UNKNOWN", "Q2_reasoning_comparison": "UNKNOWN"}
def normalize_options(raw_score: str) -> str:
    """
    TODO3: Normalize the options to A, B, C, D, E using both direct matches and prompt text patterns
    """
    score = raw_score.strip().upper()
    
    # Direct matches
    if score in ['A', 'B', 'C', 'D', 'E']:
        return score
    
    # Text-based mapping for Q1 (Score Comparison) based on prompt options:
    # A. Model A clearly better
    # B. Model A slightly better  
    # C. Both equivalent
    # D. Model B slightly better
    # E. Model B clearly better
    score_lower = score.lower()
    
    # Model A clearly better patterns
    if any(phrase in score_lower for phrase in [
        'model a clearly better', 'model a much better', 'model a significantly better',
        'a clearly better', 'clearly better a', 'model a is clearly'
    ]):
        return 'A'
    
    # Model A slightly better patterns  
    elif any(phrase in score_lower for phrase in [
        'model a slightly better', 'model a somewhat better', 'model a marginally better',
        'a slightly better', 'slightly better a', 'model a is slightly'
    ]):
        return 'B'
    
    # Both equivalent patterns
    elif any(phrase in score_lower for phrase in [
        'both equivalent', 'both equal', 'equivalent', 'same quality', 'equally good',
        'tie', 'similar', 'comparable', 'both models are equivalent'
    ]):
        return 'C'
    
    # Model B slightly better patterns
    elif any(phrase in score_lower for phrase in [
        'model b slightly better', 'model b somewhat better', 'model b marginally better', 
        'b slightly better', 'slightly better b', 'model b is slightly'
    ]):
        return 'D'
    
    # Model B clearly better patterns
    elif any(phrase in score_lower for phrase in [
        'model b clearly better', 'model b much better', 'model b significantly better',
        'b clearly better', 'clearly better b', 'model b is clearly'
    ]):
        return 'E'
    
    # Text-based mapping for Q2 (Reasoning Comparison) based on prompt options:
    # A. Model A reasoning is much better
    # B. Model A reasoning is slightly better
    # C. Both reasoning are equivalent 
    # D. Model B reasoning is slightly better
    # E. Model B reasoning is much better
    
    # Model A reasoning much better patterns
    elif any(phrase in score_lower for phrase in [
        'model a reasoning is much better', 'model a much better reasoning',
        'a reasoning much better', 'reasoning much better a'
    ]):
        return 'A'
    
    # Model A reasoning slightly better patterns
    elif any(phrase in score_lower for phrase in [
        'model a reasoning is slightly better', 'model a slightly better reasoning',
        'a reasoning slightly better', 'reasoning slightly better a'
    ]):
        return 'B'
    
    # Both reasoning equivalent patterns
    elif any(phrase in score_lower for phrase in [
        'both reasoning are equivalent', 'both reasoning equivalent', 
        'reasoning equivalent', 'reasoning are equivalent'
    ]):
        return 'C'
    
    # Model B reasoning slightly better patterns  
    elif any(phrase in score_lower for phrase in [
        'model b reasoning is slightly better', 'model b slightly better reasoning',
        'b reasoning slightly better', 'reasoning slightly better b'
    ]):
        return 'D'
    
    # Model B reasoning much better patterns
    elif any(phrase in score_lower for phrase in [
        'model b reasoning is much better', 'model b much better reasoning',
        'b reasoning much better', 'reasoning much better b'
    ]):
        return 'E'
    
    # If no match found, return UNKNOWN
    return 'UNKNOWN'

def map_option_to_model(option: str, model_a_actual: str, model_b_actual: str) -> Tuple[str, str]:
    """
    TODO4: Map options to actual model names, do not mix up
    Returns (winning_model, preference_level)
    
    Examples:
    - Option A + model_a_actual=meta -> (meta, clearly_better)
    - Option B + model_a_actual=paschalidis -> (paschalidis, slightly_better)  
    - Option C -> (both, equivalent)
    - Option D + model_b_actual=paschalidis -> (paschalidis, slightly_better)
    - Option E + model_b_actual=meta -> (meta, clearly_better)
    """
    if option == 'A':  # Model A clearly better
        return (model_a_actual, 'clearly_better')
    elif option == 'B':  # Model A slightly better
        return (model_a_actual, 'slightly_better')
    elif option == 'C':  # Both equivalent
        return ('both', 'equivalent')
    elif option == 'D':  # Model B slightly better
        return (model_b_actual, 'slightly_better')
    elif option == 'E':  # Model B clearly better
        return (model_b_actual, 'clearly_better')
    else:
        return ('unknown', 'unknown')

def process_samples(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    TODO4: Loop over samples and map options to actual model names
    """
    # Initialize results tracking
    q1_results = []  # (winning_model, preference_level)
    q2_results = []  # (winning_model, preference_level)
    
    parsing_errors = 0
    total_samples = len(data)
    
    # Process each sample
    for i, sample in enumerate(data):
        sample_id = sample.get('sample_id', i)
        model_comparison = sample.get('model_comparison', '')
        model_a_actual = sample.get('model_a_actual')
        model_b_actual = sample.get('model_b_actual')
        
        if not model_a_actual or not model_b_actual:
            parsing_errors += 1
            continue
            
        try:
            # Parse the model comparison
            parsed_comparison = parse_model_comparison(model_comparison)
            
            # Extract and normalize scores
            q1_raw = parsed_comparison.get('Q1_score_comparison', 'UNKNOWN')
            q2_raw = parsed_comparison.get('Q2_reasoning_comparison', 'UNKNOWN')
            
            q1_normalized = normalize_options(q1_raw)
            q2_normalized = normalize_options(q2_raw)
            
            # Map to actual models
            q1_winner, q1_level = map_option_to_model(q1_normalized, model_a_actual, model_b_actual)
            q2_winner, q2_level = map_option_to_model(q2_normalized, model_a_actual, model_b_actual)
            
            q1_results.append((q1_winner, q1_level))
            q2_results.append((q2_winner, q2_level))
            
        except Exception as e:
            parsing_errors += 1
            q1_results.append(('unknown', 'unknown'))
            q2_results.append(('unknown', 'unknown'))
    
    return {
        'total_samples': total_samples,
        'parsing_errors': parsing_errors,
        'q1_results': q1_results,
        'q2_results': q2_results
    }

def calculate_model_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO5: Count and calculate percentages
    """
    total_samples = results['total_samples']
    
    # Count results for each model and question
    model_counts = {
        'meta-llama': {
            'q1_clearly_better': 0,
            'q1_slightly_better': 0, 
            'q2_clearly_better': 0,
            'q2_slightly_better': 0
        },
        'Paschalidis-NOC-Lab': {
            'q1_clearly_better': 0,
            'q1_slightly_better': 0,
            'q2_clearly_better': 0, 
            'q2_slightly_better': 0
        },
        'both_equivalent': {
            'q1': 0,
            'q2': 0
        }
    }
    
    # Count Q1 results
    for winner, level in results['q1_results']:
        if winner == 'both':
            model_counts['both_equivalent']['q1'] += 1
        elif winner in model_counts and level in ['clearly_better', 'slightly_better']:
            model_counts[winner][f'q1_{level}'] += 1
    
    # Count Q2 results  
    for winner, level in results['q2_results']:
        if winner == 'both':
            model_counts['both_equivalent']['q2'] += 1
        elif winner in model_counts and level in ['clearly_better', 'slightly_better']:
            model_counts[winner][f'q2_{level}'] += 1
    
    # Calculate percentages
    model_percentages = {}
    for model, counts in model_counts.items():
        model_percentages[model] = {}
        for metric, count in counts.items():
            model_percentages[model][metric] = (count / total_samples) * 100
    
    return {
        'counts': model_counts,
        'percentages': model_percentages,
        'total_samples': total_samples
    }

def print_concise_summary(performance: Dict[str, Any]):
    """
    TODO5: Print concise summary for each question
    """
    total = performance['total_samples']
    counts = performance['counts']
    percentages = performance['percentages']
    
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"Total Samples: {total}")
    
    # Q1 Score Comparison
    print(f"\nQ1 - SCORE COMPARISON:")
    print(f"  Meta clearly better:         {counts['meta-llama']['q1_clearly_better']:3d} ({percentages['meta-llama']['q1_clearly_better']:5.2f}%)")
    print(f"  Meta slightly better:        {counts['meta-llama']['q1_slightly_better']:3d} ({percentages['meta-llama']['q1_slightly_better']:5.2f}%)")
    print(f"  Both equivalent:             {counts['both_equivalent']['q1']:3d} ({percentages['both_equivalent']['q1']:5.2f}%)")
    print(f"  Paschalidis slightly better: {counts['Paschalidis-NOC-Lab']['q1_slightly_better']:3d} ({percentages['Paschalidis-NOC-Lab']['q1_slightly_better']:5.2f}%)")
    print(f"  Paschalidis clearly better:  {counts['Paschalidis-NOC-Lab']['q1_clearly_better']:3d} ({percentages['Paschalidis-NOC-Lab']['q1_clearly_better']:5.2f}%)")
    
    # Q2 Reasoning Comparison
    print(f"\nQ2 - REASONING COMPARISON:")
    print(f"  Meta clearly better:         {counts['meta-llama']['q2_clearly_better']:3d} ({percentages['meta-llama']['q2_clearly_better']:5.2f}%)")
    print(f"  Meta slightly better:        {counts['meta-llama']['q2_slightly_better']:3d} ({percentages['meta-llama']['q2_slightly_better']:5.2f}%)")
    print(f"  Both equivalent:             {counts['both_equivalent']['q2']:3d} ({percentages['both_equivalent']['q2']:5.2f}%)")
    print(f"  Paschalidis slightly better: {counts['Paschalidis-NOC-Lab']['q2_slightly_better']:3d} ({percentages['Paschalidis-NOC-Lab']['q2_slightly_better']:5.2f}%)")
    print(f"  Paschalidis clearly better:  {counts['Paschalidis-NOC-Lab']['q2_clearly_better']:3d} ({percentages['Paschalidis-NOC-Lab']['q2_clearly_better']:5.2f}%)")

def analyze_model_comparison_results(file_path: str) -> Dict[str, Any]:
    """
    Main analysis function following the TODO workflow
    """
    # TODO1: Load JSON data
    data = load_comparison_data(file_path)
    if not data:
        return {}
    
    # TODO2-4: Process samples (parse, normalize, map to models)
    results = process_samples(data)
    
    # TODO5: Calculate percentages and performance
    performance = calculate_model_performance(results)
    
    return performance

def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM-as-Judge model comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        default="./Result/LLM_Judge/gpt-5-chat-latest/model_comparison_results.json",
        help="Input JSON file with model comparison results"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for analysis results (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        return 1
    
    print(f"Analyzing: {os.path.basename(args.input)}")
    
    # Analyze the model comparison results
    performance = analyze_model_comparison_results(args.input)
    
    if performance:
        # Print concise summary
        print_concise_summary(performance)
        
        # Save results if output specified
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(performance, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Analysis saved to: {args.output}")
            except Exception as e:
                print(f"⚠️  Could not save analysis: {e}")
                return 1
    else:
        print("❌ Analysis failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
