#!/usr/bin/env python3

import json
import re
import sys
import os
import argparse
from typing import Dict, List, Any, Tuple
from collections import Counter

# Add parent directory to path for importing utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import load_json_file, save_json_file

def parse_reference_validation(reference_validation: Any) -> Dict[str, str]:
    # If it's already a dictionary, return it
    if isinstance(reference_validation, dict):
        return reference_validation
    
    # If it's a string, try to parse as JSON first
    if isinstance(reference_validation, str):
        # Skip error messages
        if reference_validation.startswith("ERROR:"):
            return {"Q1_score_appropriateness": "UNKNOWN", "Q2_reasoning_quality": "UNKNOWN"}
        
        try:
            return json.loads(reference_validation)
        except json.JSONDecodeError:
            # If JSON parsing fails, use regex to extract values
            result = {}
            for field, pattern in [
                ("Q1_score_appropriateness", r'"Q1_score_appropriateness"\s*:\s*"([^"]*)"'),
                ("Q2_reasoning_quality", r'"Q2_reasoning_quality"\s*:\s*"([^"]*)"')
            ]:
                match = re.search(pattern, reference_validation)
                result[field] = match.group(1) if match else "UNKNOWN"
            return result
    
    # If it's neither string nor dict, return unknown
    return {"Q1_score_appropriateness": "UNKNOWN", "Q2_reasoning_quality": "UNKNOWN"}

def normalize_score(score: str) -> str:
    score = score.strip().upper()
    
    # Direct matches
    if score in ['A', 'B', 'C']:
        return score
    
    # Text-based mapping
    score_lower = score.lower()
    
    # Mapping patterns
    patterns = {
        'A': ['too low', 'underestimate', 'insufficient', 'inadequate', 'poor', 'bad', 'weak', 'flawed'],
        'B': ['appropriate', 'correct', 'suitable', 'adequate', 'reasonable', 'average', 'acceptable', 'moderate', 'fair', 'decent'],
        'C': ['too high', 'overestimate', 'excessive', 'inflated', 'good', 'excellent', 'strong', 'comprehensive', 'well-reasoned']
    }
    
    for grade, keywords in patterns.items():
        if any(word in score_lower for word in keywords):
            return grade
    
    return 'UNKNOWN'

def main():
    parser = argparse.ArgumentParser(description="Analyze LLM-as-Judge validation results")
    parser.add_argument("--input", "-i", 
                       default="./Result/LLM_Judge/gemini-2.5-pro/reference_validation_results.json",
                       help="Input JSON file with validation results")
    parser.add_argument("--output_dir", "-d", 
                       default="./Result/LLM_Judge/gemini-2.5-pro",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
    
    # Load and parse data
    try:
        data = load_json_file(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        
    
    parsed_samples = []
    errors = 0
    
    # Parse each sample
    for i, sample in enumerate(data):
        sample_id = sample.get('sample_id', i)
        try:
            parsed_validation = parse_reference_validation(sample.get('reference_validation', ''))
            q1_normalized = normalize_score(parsed_validation.get('Q1_score_appropriateness', 'UNKNOWN'))
            q2_normalized = normalize_score(parsed_validation.get('Q2_reasoning_quality', 'UNKNOWN'))
            
            parsed_samples.append({
                'sample_id': sample_id,
                'domain': sample.get('domain'),
                'q1_normalized': q1_normalized,
                'q2_normalized': q2_normalized,
            })
        except:
            errors += 1
            parsed_samples.append({
                'sample_id': sample_id,
                'domain': sample.get('domain'),
                'q1_normalized': 'UNKNOWN',
                'q2_normalized': 'UNKNOWN',
            })
    
    # Count results
    q1_counts = Counter(s['q1_normalized'] for s in parsed_samples)
    q2_counts = Counter(s['q2_normalized'] for s in parsed_samples)
    total = len(data)
    
    # Print results
    print(f"Total: {total}, Errors: {errors}")
    print(f"Q1: A={q1_counts['A']} ({q1_counts['A']/total*100:.2f}%) B={q1_counts['B']} ({q1_counts['B']/total*100:.2f}%) C={q1_counts['C']} ({q1_counts['C']/total*100:.2f}%)")
    print(f"Q2: A={q2_counts['A']} ({q2_counts['A']/total*100:.2f}%) B={q2_counts['B']} ({q2_counts['B']/total*100:.2f}%) C={q2_counts['C']} ({q2_counts['C']/total*100:.2f}%)  ")
    
    # Save files
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    save_json_file(os.path.join(args.output_dir, f"{base_name}_parsed_samples.json"), parsed_samples)

    analysis = {
        'total_samples': total,
        'parsing_errors': errors,
        'q1_counts': dict(q1_counts),
        'q2_counts': dict(q2_counts)
    }
    
    save_json_file(os.path.join(args.output_dir, f"{base_name}_analysis.json"), analysis)


if __name__ == "__main__":
    main()

