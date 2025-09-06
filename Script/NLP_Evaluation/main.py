#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from metrics_calculator import evaluate_text_generation_metrics
from data_parser import load_json, extract_information
from typing import List, Dict, Any


def filter_data_by_domain(data: List[Dict[str, Any]], target_domain: str) -> List[Dict[str, Any]]:
    """
    Filter data entries based on domain.
    
    Args:
        data: List of dict entries from JSON
        target_domain: The domain to filter for
        
    Returns:
        Filtered list of entries
    """
    # If domain is Severity or Non_severity, don't filter
    if target_domain.lower() in ['severity', 'non_severity']:
        return data
    
    # For other domains, filter based on the domain key in each entry
    filtered_data = []
    original_count = len(data)
    
    for entry in data:
        entry_domain = entry.get("domain", "").lower()
        if entry_domain == target_domain.lower():
            filtered_data.append(entry)
    
    filtered_count = len(filtered_data)
    print(f"Domain filtering summary:")
    print(f"  Target domain: {target_domain}")
    print(f"  Original entries: {original_count}")
    print(f"  Filtered entries: {filtered_count}")
    print(f"  Removed entries: {original_count - filtered_count}")
    
    return filtered_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prediction results")
    parser.add_argument("--input_file", type=str,
                        default="./Result/meta-llama/Llama-3.2-3B-Instruct/Non_severity_data/Reliability_predictions.json", help="Input JSON files")
    parser.add_argument("--org", type=str,
                        default="meta-llama", help="Organization name")
    parser.add_argument("--model", type=str,
                        default="Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--output_file", type=str,
                        default="./Result/meta-llama/Llama-3.2-3B-Instruct/Non_severity_data/Reliability.csv", help="Directory to save processed results")
    parser.add_argument("--domain", type=str,
                        default="Reliability",
                        help="Domain name to specify keys or logic: Severity, Non_severity, Clinical_impact, Pathogen, Transmission, External_risk, Geographic, Healthcare_system")
    parser.add_argument("--reasoning_keys", nargs='+', default=["Reliability_reasoning"],
                        help="List of possible reasoning keys for data extraction")
    parser.add_argument("--category_keys", nargs='+', default=["Reliability"],
                        help="List of possible category keys for data extraction")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Skipping...")
        exit(1)

    print(f"Processing: {input_file}")

    # Load JSON data
    data = load_json(input_file)
    
    # Filter data by domain
    data = filter_data_by_domain(data, args.domain)

    result = extract_information(data, args.reasoning_keys, args.category_keys)

    # Print extraction summary
    counts = result["counts"]
    print(f"Data extraction summary:")
    print(f"  Total samples: {counts['total_samples']}")
    print(f"  Successfully processed: {counts['successful_entries']}")
    print(f"  Skipped due to errors: {counts['skipped_entries']}")
    print(f"  GT categories found: {counts['gt_categories_count']}")
    print(f"  GT reasoning found: {counts['gt_reasoning_count']}")
    print(f"  Output categories found: {counts['output_categories_count']}")
    print(f"  Output reasoning found: {counts['output_reasoning_count']}")

    # Set up metric calculation parameters
    use_mae, use_mse, use_exact_match = False, False, True
    if result["GT_categories"] and isinstance(result["GT_categories"][0], (int, float)):  # Numeric category
        use_mae, use_mse = True, True

    if not result["GT_categories"] or not result["Output_categories"]:
        category_scores = {}
    else:
        category_scores = evaluate_text_generation_metrics(
            predictions=result["Output_categories"],
            references=result["GT_categories"],
            use_mae=use_mae,
            use_mse=use_mse,
            use_exact_match=use_exact_match
        )
    if not result["GT_reasoning"] or not result["Output_reasoning"]:
        reasoning_scores = {}
    else:
        reasoning_scores = evaluate_text_generation_metrics(
            predictions=result["Output_reasoning"],
            references=result["GT_reasoning"],
            use_bleu=True,
            use_rouge=True,
            use_meteor=True,
            use_bertscore=True
        )

    # Combine all scores and add model + org info + counts
    all_scores = {
        "Org": args.org,
        "Model": args.model,
        "Domain": args.domain,
        "Total_Samples": counts['total_samples'],
        "GT_Categories_Count": counts['gt_categories_count'],
        "GT_Reasoning_Count": counts['gt_reasoning_count'],
        "Output_Categories_Count": counts['output_categories_count'],
        "Output_Reasoning_Count": counts['output_reasoning_count'],
        **category_scores,
        **reasoning_scores
    }

    # Save to CSV
    df = pd.DataFrame([all_scores])
    df.to_csv(output_file, index=False)

    print("\nEvaluation Results:")
    print(df.to_string(index=False))
    print(f"\nEvaluation completed! Metrics saved at: {output_file}")
