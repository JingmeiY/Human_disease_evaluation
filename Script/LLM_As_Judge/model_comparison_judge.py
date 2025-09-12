#!/usr/bin/env python3
"""
LLM-as-Judge: Model Comparison Script

This script compares two AI model outputs (fine-tuned BEACON LLM vs base model)
for disease outbreak severity evaluation. It performs the same comparison
tasks as outlined in the Google Form for human experts.
"""

import json
import argparse
import os
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for importing utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import load_json_file, save_json_file

# OpenAI-compatible client for multiple providers
from openai import OpenAI


class ModelComparisonJudge:
    """LLM-as-Judge for comparing two model outputs"""
    
    def __init__(self, judge_model: str = "deepseek-chat", base_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the judge with specified model and provider"""
        self.judge_model = judge_model
        
        # Configure client based on model provider
        if "deepseek" in judge_model.lower():
            self.client = OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url or "https://api.deepseek.com"
            )
        elif "gemini" in judge_model.lower():
            self.client = OpenAI(
                api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            # Default to OpenAI or custom provider
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url
            )
    
    def create_comparison_prompt(self, sample: Dict[str, Any]) -> str:
        """Create the comparison prompt for a single sample"""
        
        prompt = f"""
You are given an article, the domain-specific guidelines used for assessments, and two model assessments to compare.

## INSTRUCTIONS

1. **Analyze the Article:** First, carefully read the article to identify the key facts, epidemiological information, and potential public health risks.

2. **Evaluate Both Assessments:** Carefully compare both Model A and Model B assessments against the article's content and assessment guidelines. When articles contain limited information, apply your domain expertise in epidemiology and infectious disease principles to evaluate which assessment more appropriately draws upon established scientific knowledge to assign appropriate scores and provide comprehensive reasoning.

3. **Compare and Formulate Answers:** Based on your expert analysis, determine which model provides a more appropriate severity score and better reasoning quality. 

### 1. ARTICLE CONTENT
{sample.get('article_content', 'Article content not available')}

### 2. ASSESSMENT GUIDELINES
The following assessments were generated based on the '{sample.get('domain', 'Unknown')}' domain, using these guidelines:
{sample.get('factors_description', 'Not specified')}

### 3. SEVERITY SCALE
1 = Very Low Risk, 2 = Low Risk, 3 = Moderate Risk, 4 = High Risk, 5 = Very High Risk

## EVALUATION QUESTIONS

**Q1: Score Comparison**

Model A Score: {sample.get('model_a_score', 'N/A')}/5

Model B Score: {sample.get('model_b_score', 'N/A')}/5

Which model provides a more appropriate severity score for this scenario?

A. Model A clearly better
B. Model A slightly better  
C. Both equivalent
D. Model B slightly better
E. Model B clearly better

**Q2: Reasoning Comparison**

Model A Reasoning: {sample.get('model_a_reasoning', 'No reasoning provided')}

Model B Reasoning: {sample.get('model_b_reasoning', 'No reasoning provided')}

Which model provides better reasoning and justification?

A. Model A reasoning is much better
B. Model A reasoning is slightly better
C. Both reasoning are equivalent 
D. Model B reasoning is slightly better
E. Model B reasoning is much better

## REQUIRED OUTPUT FORMAT

Your response MUST be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.

{{"Q1_score_comparison": "A, B, C, D, or E", "Q1_explanation": "Explain which model's score is more appropriate and why. Be concise (1-2 sentences).", "Q2_reasoning_comparison": "A, B, C, D, or E", "Q2_explanation": "Explain which model's reasoning is better, pointing out specific strengths or weaknesses. Be concise (1-2 sentences)."}}
"""
        return prompt
    
    def compare_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model outputs for a single sample"""
        
        sample_id = sample.get('sample_id', 'unknown')
        print(f"Comparing models for sample {sample_id}...")
        
        prompt = self.create_comparison_prompt(sample)
        print(f"Prompt:\n\n {prompt}\n\n")
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert epidemiologist and public health specialist conducting systematic comparisons of AI-generated disease outbreak severity assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            
            # Save raw response content directly
            raw_response = response.choices[0].message.content
            print(f"Raw response: {raw_response}\n\n")
            
            # Create result with original sample data + comparison
            result = {
                **sample,  # Include all original fields
                'model_comparison': raw_response
            }
            
            return result
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            
            # If error, save error message as string
            result = {
                **sample,
                'model_comparison': f"ERROR: {str(e)}"
            }
            
            return result
    
    def compare_all_samples(self, samples: List[Dict[str, Any]], delay_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Compare all samples"""
        
        results = []
        total_samples = len(samples)
        
        print(f"Starting comparison of {total_samples} samples with {self.judge_model}")
        
        for i, sample in enumerate(samples, 1):
            sample_id = sample.get('sample_id', f'sample_{i}')
            print(f"Progress: {i}/{total_samples} - Sample {sample_id}")
            
            result = self.compare_single_sample(sample)
            results.append(result)
            
            # Add delay to avoid rate limiting
            if i < total_samples and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        print(f"Completed comparison of {total_samples} samples")
        return results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Model Comparison")
    parser.add_argument("--judge_model", default="deepseek-reasoner", help="Judge model name")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--input_file", default="./Result/evaluation_data/selected_evaluation_samples.json", help="Path to input JSON file with samples")
    parser.add_argument("--output_dir", default="./Result/LLM_Judge/deepseek-reasoner", help="Path to input JSON file with samples")
    parser.add_argument("--judge_config_path", default="./llm_judge_config.json", help="Path to judge config file")

    args = parser.parse_args()

    # Load model configuration automatically
    config = load_json_file(args.judge_config_path)[args.judge_model]

    # Load samples
    print(f"Loading samples from {args.input_file}")
    samples = load_json_file(args.input_file)
    print(f"Loaded {len(samples)} samples")
    
    # Initialize judge with auto-loaded config
    judge = ModelComparisonJudge(
        judge_model=args.judge_model,
        base_url=config["base_url"],
        api_key=config["api_key"]
    )
    
    # Compare samples
    results = judge.compare_all_samples(samples, delay_seconds=args.delay)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "model_comparison_results.json")
    save_json_file(output_file, results)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
