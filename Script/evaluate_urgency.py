import math
from openai import OpenAI
import os
from typing import List, Optional, Any, Dict, Optional
from pydantic import BaseModel, ValidationError
from datetime import datetime
import json
import re
import argparse
from utility import load_json_file, save_json_file, is_vllm_available, load_file_content
import traceback
class UrgencyRating(BaseModel):
    """
    A Pydantic model representing the urgency assessment of a given signal.

    Attributes:
        Urgency (str): The urgency level of the signal.
        Urgency_reasoning (str): Explanation or reasoning behind the urgency rating.
    """
    Urgency: str
    Urgency_reasoning: str


def evaluate_urgency(text, llm_config, system_message_path="./Resource/system_message.txt", urgency_prompt_path="./Resource/urgency_prompt.txt"):
    """
    Args:
        text (str): The text content of the signal to be evaluated.
        system_message_path (str, optional): Path to the system message file. Defaults to "system_message.txt".
        urgency_prompt_path (str, optional): Path to the urgency prompt file. Defaults to "urgency_prompt.txt".

    Returns:
        dict: A dictionary containing the urgency rating and the reasoning behind it.
    """
    system_message = load_file_content(system_message_path)
    urgency_prompt = load_file_content(urgency_prompt_path)
    client = OpenAI(base_url=llm_config["base_url"], api_key=llm_config["api_key"])

    valid_rating = ["Critical", "High", "Moderate", "Low", "Unknown"]

    try:
        completion = client.beta.chat.completions.parse(
            model=llm_config["model"]["NonSeverity"],
            temperature=0.0,
            top_p=0.9,
            max_tokens=3096,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": f"{urgency_prompt}\n{text}"}
            ],
            response_format=UrgencyRating,
        )

        response = completion.choices[0].message.parsed.model_dump()

        # If Urgency not in the valid list, set it to "Unknown"
        if response["Urgency"] not in valid_rating:
            response["Urgency"] = "Unknown"

    except Exception as e:
        error_details = traceback.format_exc()
        response = {
            "Urgency": "Unknown",
            "Urgency_reasoning": f"An error occurred: {e}\nDetails:\n{error_details}"
        }

    return response

def format_urgency_input(result):
    """
    Formats the input text for the evaluate_urgency function.

    Args:
        result (dict): A dictionary containing the signal information with keys:
            - 'Event_date' (str): The date of the event.
            - 'Recency' (str): The recency category of the event.
            - 'Signal_source_language' (str): The content of the signal in its original language.
            - 'Signal_en' (str): The content of the signal translated into English.

    Returns:
        str: A formatted string that combines the event date, recency, content, and translated signal.
    """
    event_date = result.get("Event_date", "Unknown")
    recency = result.get("Recency", "Unknown")
    content = result.get("Signal_en", "") or result.get("Signal_source_language", "")


    return (
        f"Event_date: {event_date}\n"
        f"Recency: {recency}\n"
        f"Content: {content}\n"
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate the urgency.")
    parser.add_argument("--input_json", type=str, default="./Data/Processed/Non_severity_data/test.json", help="Path to the input JSON file")
    parser.add_argument("--output_json", type=str, default="./Result/MODEL_NAME/Non_severity_data/Urgency_predictions.json", help="Path to the output JSON file")
    parser.add_argument("--config", type=str, default="model_config.json", help="Path to the config file (default: config.json)") 
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to run the evaluation function (-1 for all)")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    config = load_json_file(args.config)
    llm_config = config["vllm"] if is_vllm_available(config["vllm"]["base_url"]) else config["openai"]
    print(f"Selected LLM Service: {llm_config['base_url']}")

    data = load_json_file(args.input_json)
    # Filter for the data, only keep the entries with domain as "Urgency"
    data = [item for item in data if item.get("domain", "").lower() == "urgency"]
    print(f"Filtered data to {len(data)} entries with domain 'Urgency'")
    
    evaluated_results = []

    num_samples = len(data) if args.samples == -1 else min(args.samples, len(data))
    for ix, item in enumerate(data[:num_samples]):

        print(f"Evaluating: {ix+1}/{len(data)}")
        output = evaluate_urgency(text=item["input"], llm_config=llm_config)
        print(f"Output: {output}")
        item["output"] = json.dumps(output)
        evaluated_results.append(item)
        

    save_json_file(args.output_json, evaluated_results)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()