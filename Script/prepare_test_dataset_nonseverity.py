import json
import re
import argparse
from collections import defaultdict
import os
def load_test_json(json_path):
    """
    Load the test.json file from the given path
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully loaded {len(data)} entries from {json_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return None

def assign_domain(output_text):
    """
    Assign domain based on the first factor in the output
    """
    if "Quality_reasoning" in output_text:
        return "Quality"
    elif "Reliability_reasoning" in output_text:
        return "Reliability"
    elif "Urgency_reasoning" in output_text:
        return "Urgency"
    else:
        return "Unknown"

def extract_first_sentence(text):
    """
    Extract the first sentence from the text
    """
    # Split by common sentence endings (., !, ?)
    sentences = re.split(r'[.!?]', text)
    if sentences:
        return sentences[0].strip()
    return None

def extract_signal_start(input_text):
    """
    Extract content after 'Signal in English:' and return the first sentence
    """
    # Split by Signal in English
    signal_in_english = re.split(r"Content:", input_text)
    if len(signal_in_english) > 1:
        content = signal_in_english[1].strip()
        return extract_first_sentence(content)
    return None

def extract_article_content(input_text):
    """
    Extract article content after 'Signal in English:'
    """
    parts = input_text.split("Content:")
    if len(parts) > 1:
        return parts[1].strip()
    return input_text  # fallback to original text if pattern not found

def process_test_dataset(json_path):
    """
    Main function to process the test dataset according to requirements
    """
    # Step 1: Load the test.json
    data = load_test_json(json_path)
    if data is None:
        return None
    
    # Step 2: Process each entry
    processed_data = []
    article_id_map = {}
    current_article_id = 1
    
    for i, entry in enumerate(data):
        # Assign sample ID
        sample_id = i + 1
        
        # Assign domain based on output content
        domain = assign_domain(entry.get("output", ""))
        
        # Extract signal start for article ID assignment
        input_text = entry.get("input", "")
        signal_start = extract_signal_start(input_text)
        
        # Assign article ID based on signal start
        if signal_start:
            if signal_start not in article_id_map:
                article_id_map[signal_start] = current_article_id
                current_article_id += 1
            article_id = article_id_map[signal_start]
        else:
            # Assign a unique article ID for entries without "Content:" pattern
            article_id = current_article_id
            current_article_id += 1
        
        # Extract article content
        article_content = extract_article_content(input_text)
        
        # Create processed entry
        processed_entry = {
            "sample_id": sample_id,
            "domain": domain,
            "article_id": article_id,
            "article_content": article_content,
            "instruction": entry.get("instruction", ""),
            "input": entry.get("input", ""),
            "GT": entry.get("output", "")  
        }
        
        processed_data.append(processed_entry)
    
    return processed_data

def save_processed_data(processed_data, output_path):
    """
    Save the processed data to a new JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, indent=2, ensure_ascii=False)
        print(f"Successfully saved processed data to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def analyze_dataset(processed_data):
    """
    Print analysis of the processed dataset
    """
    print("\n=== Dataset Analysis ===")
    print(f"Total entries: {len(processed_data)}")
    
    # Domain distribution
    domain_counts = defaultdict(int)
    for entry in processed_data:
        domain_counts[entry["domain"]] += 1
    
    print("\nDomain distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")
    
    # Article ID distribution
    article_ids = [entry["article_id"] for entry in processed_data if entry["article_id"] is not None]
    unique_articles = len(set(article_ids))
    print(f"\nUnique articles: {unique_articles}")
    print(f"Entries with article IDs: {len(article_ids)}")

# Main execution
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process test dataset JSON file")
    parser.add_argument(
        "--input_path", 
        default="./Data/Merged/Non_severity_data/test.json",
        help="Path to the input test.json file"
    )
    parser.add_argument(
        "--output_path", 
        default="./Data/Processed/Non_severity_data/test.json",
        help="Path to save the processed test.json file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory (but not the filename part)
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Check if output path already exists as a directory
    if os.path.isdir(args.output_path):
        print(f"Error: Output path '{args.output_path}' is a directory, not a file!")

    
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    
    # Process the dataset
    processed_data = process_test_dataset(args.input_path)
    
    if processed_data:
        # Analyze the dataset
        analyze_dataset(processed_data)
        
        # Save processed data
        save_processed_data(processed_data, args.output_path)
        
        # Show a sample entry
        print("\n=== Sample Entry ===")
        print(json.dumps(processed_data[0], indent=2))