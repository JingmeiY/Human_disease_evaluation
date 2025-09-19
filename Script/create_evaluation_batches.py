#!/usr/bin/env python3

import os
import sys
from typing import Dict, List, Any

# Add utility functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utility import load_json_file, save_json_file


def create_evaluation_batches(selected_data: List[Dict[str, Any]], 
                            batch_size: int = 5) -> List[List[Dict[str, Any]]]:

    batches = []
    
    # Create batches using step size
    for i in range(0, len(selected_data), batch_size):
        batch = selected_data[i:i + batch_size]
        batches.append(batch)
    
    return batches


def main():

    base_dir = "./"
    input_path = os.path.join(base_dir, "Result/evaluation_data/selected_evaluation_samples.json")
    output_dir = os.path.join(base_dir, "Result/evaluation_data/batches")
    
    os.makedirs(output_dir, exist_ok=True)
    
    selected_data = load_json_file(input_path)
    
    batch_size = 2  
    batches = create_evaluation_batches(selected_data, batch_size=batch_size)
    
    for i, batch in enumerate(batches):
        batch_id = i + 1
        batch_filename = f"evaluation_samples_batch_{batch_id}.json"
        batch_path = os.path.join(output_dir, batch_filename)
        save_json_file(batch_path, batch)
    


if __name__ == "__main__":
    main()
