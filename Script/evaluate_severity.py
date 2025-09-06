import math
from openai import OpenAI
import os
from typing import List, Optional, Any, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import re
import argparse
from utility import load_json_file, save_json_file, is_vllm_available, load_file_content
import traceback
class IndividualFactor(BaseModel):
    Factor: str
    Score: int
    Reasoning: str


class SeverityAssessment(BaseModel):
    Individual_assessments: List[IndividualFactor]
    Overall_reasoning: str
    Overall_score: int


def evaluate_severity(text, llm_config, model, system_message_path="./Resource/system_message.txt", severity_prompt_path="./Resource/severity_prompt.txt"):
    system_message = load_file_content(system_message_path)
    severity_prompt = load_file_content(severity_prompt_path)
    client = OpenAI(base_url=llm_config["base_url"], api_key=llm_config["api_key"])
    valid_rating = [1, 2, 3, 4, 5]

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            temperature=0.0,
            top_p=0.9,
            max_tokens=3096,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{severity_prompt}\n{text}"}
            ],
            response_format=SeverityAssessment,
        )
        response = completion.choices[0].message.parsed.model_dump()

        # If Overall_score not in the valid list, set it to "Unknown"
        if int(response["Overall_score"]) not in valid_rating:
            response["Overall_score"] = 0

    except Exception as e:
        error_details = traceback.format_exc()
        response = {
            "Individual_assessments": [],
            "Overall_score": 0,
            "Overall_reasoning": f"An error occurred: {e}\nDetails:\n{error_details}"
        }

    return response


def format_severity_input(result, rag_system):
    external_resource = rag_system.get_risk(result['Diseases'])
    content = result.get("Signal_en", "") or result.get("Signal_source_language", "")

    return (
        f"Pathogen characteristics: {external_resource}\n"
        f"Content: {content}\n"
    )



def compute_weighted_severity(severity_responses, weights):
    # Extract overall scores for each factor and round up to int
    scores = {}
    for factor_name, response in severity_responses.items():
        overall_score = response.get('Overall_score', 0)
        scores[factor_name] = int(math.ceil(overall_score))

    # Calculate the weighted average
    total_weight = sum(weights.values())
    weighted_sum = sum(scores[factor] * weights.get(factor, 1) for factor in scores)
    weighted_average = weighted_sum / total_weight if total_weight else 0
    weighted_average = int(math.ceil(weighted_average))
    return weighted_average, scores

def calculate_severity(result, factors, llm_config, weights, rag_system):
    severity_responses = {}

    # Iterate over the factors and get the severity responses
    for factor in factors:
        severity_response = evaluate_severity(
            text=format_severity_input(result, rag_system),
            llm_config=llm_config, model=llm_config["model"][factor], severity_prompt_path=factors[factor]
        )
        severity_responses[factor] = severity_response

    # Compute the weighted severity
    weighted_severity, individual_scores = compute_weighted_severity(severity_responses, weights)

    severity_result = {
        'Severity_reasoning': severity_responses,
        'Individual_scores': individual_scores,
        'Severity_score': weighted_severity
    }

    return severity_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the urgency.")
    parser.add_argument("--input_json", type=str, default="./test.json", help="Path to the input JSON file")
    parser.add_argument("--output_json", type=str, default="./predictions.json", help="Path to the output JSON file")
    parser.add_argument("--config", type=str, default="model_config.json", help="Path to the config file (default: config.json)")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to run the evaluation function (-1 for all)")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    config = load_json_file(args.config)
    print(f"Config: {config}")
    llm_config = config["vllm"] if is_vllm_available(config["vllm"]["base_url"]) else config["openai"]
    print(f"Selected LLM Service: {llm_config['base_url']}")
    factors = load_json_file('./Resource/severity_factors.json')
    

    data = load_json_file(args.input_json)
    evaluated_results = []

    num_samples = len(data) if args.samples == -1 else min(args.samples, len(data))
    for ix, item in enumerate(data[:num_samples]): 
        print(f"Evaluating: {ix+1}/{len(data)}")
        factor = item["domain"]
        output = evaluate_severity(text=item["input"], llm_config=llm_config, model=llm_config["model"][factor], severity_prompt_path=factors[factor])
        print(f"Output: {output}")
        item["output"] = json.dumps(output)
        evaluated_results.append(item)
        

    save_json_file(args.output_json, evaluated_results)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()