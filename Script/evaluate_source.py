from openai import OpenAI
from pydantic import BaseModel
import os
from typing import List, Optional, Any, Dict, Optional
from datetime import datetime
import json
import re
import argparse
from utility import load_json_file, save_json_file, is_vllm_available, load_file_content
import traceback

class ReliabilityIndex(BaseModel):
    """
    A Pydantic model representing the reliability assessment of a given text.

    Attributes:
        Reliability (str): The assessed reliability level of the text.
        Reliability_reasoning (str): Explanation or reasoning behind the reliability assessment.
    """
    Reliability: str
    Reliability_reasoning: str

def evaluate_reliability(text, llm_config, source_system_path='./Resource/source_system.txt', source_prompt_path='./Resource/source_prompt.txt'):
    """
    Evaluates the reliability of the given text.

    Args:
        text (str): The text content to be evaluated for reliability.
        source_system_path (str, optional): Path to the system prompt file. Defaults to 'source_system.txt'.
        source_prompt_path (str, optional): Path to the user prompt file. Defaults to 'source_prompt.txt'.

    Returns:
        dict: A dictionary containing the reliability assessment and the reasoning behind it.
    """
    source_system = load_file_content(source_system_path)
    source_prompt = load_file_content(source_prompt_path)
    client = OpenAI(base_url=llm_config["base_url"], api_key=llm_config["api_key"])

    valid_rating = ["Highly Reliable", "Very Reliable", "Moderately Reliable", "Reliable", "Variable Reliability", "Unverified", "Unknown"]
    try:
        completion = client.beta.chat.completions.parse(
            model=llm_config["model"]["NonSeverity"],
            temperature=0.0,
            top_p=0.9,
            max_tokens=3096,
            messages=[
                {"role": "system", "content": source_system},
                {"role": "user", "content":  f"{source_prompt}\n{text}"}
            ],
            response_format=ReliabilityIndex,
        )
        response = completion.choices[0].message.parsed.model_dump()
        

        # If Reliability not in the valid list, set it to "Unknown"
        if response["Reliability"] not in valid_rating:
            response["Reliability"] = "Unknown"

    except Exception as e:
        error_details = traceback.format_exc()
        response = {
            "Reliability": "Unknown",
            "Reliability_reasoning": f"An error occurred: {e}\nDetails:\n{error_details}"
        }

    return response

def main():
    parser = argparse.ArgumentParser(description="Evaluate the source reliability.")
    parser.add_argument("--input_json", type=str, default="./Data/Processed/Non_severity_data/test.json", help="Path to the input JSON file")
    parser.add_argument("--output_json", type=str, default="./Result/MODEL_NAME/Non_severity_data/Reliability_predictions.json", help="Path to the output JSON file")
    parser.add_argument("--config", type=str, default="model_config.json", help="Path to the config file (default: config.json)") 
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to run the evaluation function (-1 for all)")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    config = load_json_file(args.config)
    llm_config = config["vllm"] if is_vllm_available(config["vllm"]["base_url"]) else config["openai"]
    print(f"Selected LLM Service: {llm_config['base_url']}")

    data = load_json_file(args.input_json)
    # Filter for the data, only keep the entries with domain as "Reliability"
    data = [item for item in data if item.get("domain", "").lower() == "reliability"]
    print(f"Filtered data to {len(data)} entries with domain 'Reliability'")
    evaluated_results = []

    num_samples = len(data) if args.samples == -1 else min(args.samples, len(data))
    for ix, item in enumerate(data[:num_samples]): 
        print(f"Evaluating: {ix+1}/{len(data)}")
        output = evaluate_reliability(text=item["input"], llm_config=llm_config)
        print(f"Output: {output}")
        item["output"] = json.dumps(output)
        evaluated_results.append(item)
        

    save_json_file(args.output_json, evaluated_results)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()