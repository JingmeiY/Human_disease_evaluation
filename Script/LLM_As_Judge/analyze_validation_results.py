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
from utility import load_json_file

def parse_reference_validation(reference_validation: Any) -> Dict[str, str]:
    """
    Parse the reference_validation field which can be:
    - A valid JSON string
    - A dictionary
    - An invalid JSON string (parse with regex)
    - An error string
    """
    
    # If it's already a dictionary, return it
    if isinstance(reference_validation, dict):
        return reference_validation
    
    # If it's a string, try to parse as JSON first
    if isinstance(reference_validation, str):
        # Skip error messages
        if reference_validation.startswith("ERROR:"):
            return {"Q1_score_appropriateness": "UNKNOWN", "Q2_reasoning_quality": "UNKNOWN"}
        
        try:
            # Try to parse as JSON
            parsed = json.loads(reference_validation)
            return parsed
        except json.JSONDecodeError:
            # If JSON parsing fails, use regex to extract values
            result = {}
            
            # Extract Q1_score_appropriateness
            q1_match = re.search(r'"Q1_score_appropriateness"\s*:\s*"([^"]*)"', reference_validation)
            if q1_match:
                result["Q1_score_appropriateness"] = q1_match.group(1)
            else:
                result["Q1_score_appropriateness"] = "UNKNOWN"
            
            # Extract Q2_reasoning_quality
            q2_match = re.search(r'"Q2_reasoning_quality"\s*:\s*"([^"]*)"', reference_validation)
            if q2_match:
                result["Q2_reasoning_quality"] = q2_match.group(1)
            else:
                result["Q2_reasoning_quality"] = "UNKNOWN"
            
            return result
    
    # If it's neither string nor dict, return unknown
    return {"Q1_score_appropriateness": "UNKNOWN", "Q2_reasoning_quality": "UNKNOWN"}

def normalize_score(score: str) -> str:
    """
    Normalize LLM output to A, B, or C categories
    """
    score = score.strip().upper()
    
    # Direct matches
    if score in ['A', 'B', 'C']:
        return score
    
    # Text-based mapping for Q1 (Score Appropriateness)
    if any(word in score.lower() for word in ['too low', 'underestimate', 'insufficient', 'inadequate']):
        return 'A'
    elif any(word in score.lower() for word in ['appropriate', 'correct', 'suitable', 'adequate', 'reasonable']):
        return 'B'
    elif any(word in score.lower() for word in ['too high', 'overestimate', 'excessive', 'inflated']):
        return 'C'
    
    # Text-based mapping for Q2 (Reasoning Quality)
    if any(word in score.lower() for word in ['poor', 'bad', 'weak', 'flawed', 'inadequate']):
        return 'A'
    elif any(word in score.lower() for word in ['average', 'acceptable', 'moderate', 'fair', 'decent']):
        return 'B'
    elif any(word in score.lower() for word in ['good', 'excellent', 'strong', 'comprehensive', 'well-reasoned']):
        return 'C'
    
    # If no match found, return UNKNOWN
    return 'UNKNOWN'

def analyze_validation_results(file_path: str) -> Dict[str, Any]:
    """
    Main function to analyze validation results
    """
    
    try:
        data = load_json_file(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}
    
    # Initialize counters
    q1_scores = []  # Score appropriateness
    q2_scores = []  # Reasoning quality
    
    parsing_errors = 0
    total_samples = len(data)
    
    # Parse each sample's reference_validation
    for i, sample in enumerate(data):
        sample_id = sample.get('sample_id', i)
        reference_validation = sample.get('reference_validation', '')
        
        # Parse the reference validation
        try:
            parsed_validation = parse_reference_validation(reference_validation)
            
            # Extract and normalize scores
            q1_raw = parsed_validation.get('Q1_score_appropriateness', 'UNKNOWN')
            q2_raw = parsed_validation.get('Q2_reasoning_quality', 'UNKNOWN')
            
            q1_normalized = normalize_score(q1_raw)
            q2_normalized = normalize_score(q2_raw)
            
            q1_scores.append(q1_normalized)
            q2_scores.append(q2_normalized)
            
        except Exception as e:
            parsing_errors += 1
            q1_scores.append('UNKNOWN')
            q2_scores.append('UNKNOWN')
    
    # Count and calculate percentages
    q1_counter = Counter(q1_scores)
    q2_counter = Counter(q2_scores)
    
    def calculate_percentages(counter: Counter, total: int) -> Dict[str, float]:
        """Calculate percentages from counter"""
        percentages = {}
        for key, count in counter.items():
            percentages[key] = (count / total) * 100
        return percentages
    
    q1_percentages = calculate_percentages(q1_counter, total_samples)
    q2_percentages = calculate_percentages(q2_counter, total_samples)
    
    # Prepare summary results
    results = {
        'total_samples': total_samples,
        'parsing_errors': parsing_errors,
        'q1_score_appropriateness': {
            'counts': dict(q1_counter),
            'percentages': q1_percentages,
            'categories': {
                'A': 'Too Low (underestimates risk)',
                'B': 'Appropriate (correct risk level)', 
                'C': 'Too High (overestimates risk)'
            }
        },
        'q2_reasoning_quality': {
            'counts': dict(q2_counter),
            'percentages': q2_percentages,
            'categories': {
                'A': 'Poor (major errors/gaps)',
                'B': 'Average (acceptable but improvable)',
                'C': 'Good (accurate and comprehensive)'
            }
        }
    }
    
    return results

def print_summary(results: Dict[str, Any]):
    """
    Print a concise summary of the analysis results
    """
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"Total Samples: {results['total_samples']} | Parsing Errors: {results['parsing_errors']}")
    
    # Q1 Score Appropriateness Analysis
    print(f"\nQ1: SCORE APPROPRIATENESS")
    print("-" * 30)
    
    q1_data = results['q1_score_appropriateness']
    for code in ['A', 'B', 'C']:
        if code in q1_data['counts']:
            count = q1_data['counts'][code]
            percentage = q1_data['percentages'][code]
            description = q1_data['categories'][code].split(' (')[0]  # Short description
            print(f"  {code}: {count:4d} ({percentage:5.2f}%) - {description}")
    
    # Find winner for Q1
    valid_q1 = {k: v for k, v in q1_data['counts'].items() if k != 'UNKNOWN'}
    if valid_q1:
        q1_winner = max(valid_q1, key=valid_q1.get)
        print(f"  → Most common: {q1_winner} ({q1_data['percentages'][q1_winner]:.2f}%)")
    
    # Q2 Reasoning Quality Analysis
    print(f"\nQ2: REASONING QUALITY")
    print("-" * 30)
    
    q2_data = results['q2_reasoning_quality']
    for code in ['A', 'B', 'C']:
        if code in q2_data['counts']:
            count = q2_data['counts'][code]
            percentage = q2_data['percentages'][code]
            description = q2_data['categories'][code].split(' (')[0]  # Short description
            print(f"  {code}: {count:4d} ({percentage:5.2f}%) - {description}")
    
    # Find winner for Q2
    valid_q2 = {k: v for k, v in q2_data['counts'].items() if k != 'UNKNOWN'}
    if valid_q2:
        q2_winner = max(valid_q2, key=valid_q2.get)
        print(f"  → Most common: {q2_winner} ({q2_data['percentages'][q2_winner]:.2f}%)")
    
    # Key Insights
    if valid_q1 and valid_q2:
        appropriate_scores = q1_data['percentages'].get('B', 0)
        good_reasoning = q2_data['percentages'].get('C', 0)
        
        print(f"\nKEY INSIGHTS:")
        print(f"• {appropriate_scores:.2f}% of scores are appropriately calibrated")
        print(f"• {good_reasoning:.2f}% of reasoning is high quality")
        
        if appropriate_scores > 50:
            print("• ✅ Majority of scores are well-calibrated")
        else:
            print("• ⚠️  Score calibration needs improvement")
            
        if good_reasoning > 50:
            print("• ✅ Majority of reasoning is comprehensive")
        else:
            print("• ⚠️  Reasoning quality could be improved")

def main():

    
    parser = argparse.ArgumentParser(
        description="Analyze LLM-as-Judge validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Essential arguments only
    parser.add_argument(
        "--input", "-i",
        default="./Result/LLM_Judge/gemini-2.5-pro/reference_validation_results.json",
        help="Input JSON file with validation results"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="reference_validation_analysis.json",
        help="Output JSON file for analysis results (optional)"
    )
    
    parser.add_argument(
        "--output_dir", "-d",
        default="./Result/LLM_Judge/gemini-2.5-pro",

        help="Output directory (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        return 1
    
    print(f"Analyzing: {os.path.basename(args.input)}")
    
    # Analyze the validation results
    results = analyze_validation_results(args.input)
    
    if results:
        # Print the summary
        print_summary(results)
        
        # Save results if output specified
        if args.output or args.output_dir:
            if args.output:
                output_file = args.output
            else:
                base_name = os.path.splitext(os.path.basename(args.input))[0]
                output_file = f"{base_name}_analysis.json"
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(args.output_dir, os.path.basename(output_file))
            elif not args.output:
                input_dir = os.path.dirname(args.input)
                output_file = os.path.join(input_dir, output_file)
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Analysis saved to: {output_file}")
            except Exception as e:
                print(f"⚠️  Could not save analysis: {e}")
                return 1
    else:
        print("❌ Analysis failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
