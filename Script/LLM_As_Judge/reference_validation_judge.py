#!/usr/bin/env python3

import json
import argparse
import os
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel

# Add parent directory to path for importing utility functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility import load_json_file, save_json_file

# OpenAI-compatible client for multiple providers
from openai import OpenAI



class ReferenceValidationJudge:
    
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
    
    def create_validation_prompt(self, sample: Dict[str, Any]) -> str:
        
        prompt = f"""
You are given an article, the domain-specific guidelines used for an assessment, and the assessment itself (a score and reasoning).

## INSTRUCTIONS
1.  **Analyze the Article:** First, carefully read the article to identify the key facts, epidemiological information, and potential public health risks.
2. **Evaluate the Assessment:** Carefully compare the provided 'Severity Score' and 'Reasoning' against the article's content and assessment guidelines. When articles contain limited information, apply your domain expertise in epidemiology and infectious disease principles to evaluate whether the overall assessment appropriately draws upon established scientific knowledge to assign appropriate scores and provide comprehensive reasoning, even when this knowledge extends beyond the specific details provided in the article.
3.  **Formulate Answers:** Based on your expert analysis, determine if the score is appropriate and if the reasoning is of high quality. Assess whether the reasoning is scientifically accurate, coherent, and complete.

### 1. ARTICLE CONTENT
{sample.get('article_content', 'Article content not available')}

### 2. ASSESSMENT GUIDELINES
The following assessment was generated based on the '{sample.get('domain', 'Unknown')}' domain, using these guidelines:
{sample.get('factors_description', 'Not specified')}

### 3. SEVERITY SCALE
1 = Very Low Risk, 2 = Low Risk, 3 = Moderate Risk, 4 = High Risk, 5 = Very High Risk

## ASSESSMENT TO EVALUATE
**Severity Score:** {sample.get('reference_score', 'N/A')}/5
**Reasoning:** {sample.get('reference_reasoning', 'No reasoning provided')}


## EVALUATION QUESTIONS

**Q1: Score Appropriateness**
Is the severity score appropriate for this scenario?

A. Too low (the score underestimates the actual risk level)
B. Appropriate (the score correctly reflects the risk level)
C. Too high (the score overestimates the actual risk level)

**Q2: Reasoning Quality**
Rate the quality of the reasoning provided:

A. Poor (major errors, significant gaps, or fundamentally flawed logic)
B. Average (acceptable but could be more comprehensive or precise)
C. Good (accurate, well-reasoned, and appropriately comprehensive)

## REQUIRED OUTPUT FORMAT
Your response MUST be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.
{{"Q1_score_appropriateness": "A, B, or C", "Q1_explanation": "Explain why the Score is Too low, Appropriate, or Too high. Be concise (1-2 sentences).", "Q2_reasoning_quality": "A, B, or C", "Q2_explanation": "Explain why the reasoning is Poor, Average, or Good, pointing out specific strengths or weaknesses. Be concise (1-2 sentences)."}}
"""
        return prompt
    
    def validate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single sample and return structured output"""
        
        sample_id = sample.get('sample_id', 'unknown')
        print(f"Validating sample {sample_id}...")
        
        prompt = self.create_validation_prompt(sample)
        print(f"Prompt: {prompt}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert public health analyst and epidemiologist with extensive experience in infectious disease surveillance, outbreak investigation, and risk assessment. Your task is to perform a rigorous, evidence-based evaluation of AI-generated severity assessments for infectious disease reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            
            # Save raw response content directly
            raw_response = response.choices[0].message.content
            print(f"Raw response: {raw_response}\n\n")
            
            # Create result with original sample data + evaluation
            result = {
                **sample,  # Include all original fields
                'reference_validation': raw_response
            }
            
            return result
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            
            # If error, save error message as string
            result = {
                **sample,
                'reference_validation': f"ERROR: {str(e)}"
            }
            
            return result
    
    def validate_all_samples(self, samples: List[Dict[str, Any]], delay_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Validate all samples"""
        
        results = []
        total_samples = len(samples)
        
        print(f"Starting validation of {total_samples} samples with {self.judge_model}")
        
        for i, sample in enumerate(samples, 1):
            sample_id = sample.get('sample_id', f'sample_{i}')
            print(f"Progress: {i}/{total_samples} - Sample {sample_id}")
            
            result = self.validate_single_sample(sample)
            results.append(result)
            
            # Add delay to avoid rate limiting
            if i < total_samples and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        print(f"Completed validation of {total_samples} samples")
        return results




def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Reference Validation")
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="Judge model name")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between API calls (seconds)")
    parser.add_argument("--input_file", default="./Result/evaluation_data/selected_evaluation_samples.json", help="Path to input JSON file with samples")
    parser.add_argument("--output_dir", default="./Result/LLM_Judge/gemini-2.5-pro", help="Path to input JSON file with samples")
    parser.add_argument("--judge_config_path", default="./llm_judge_config.json", help="Path to judge config file")

    args = parser.parse_args()

    # Load model configuration automatically
    config = load_json_file(args.judge_config_path)[args.judge_model]

    
    # Load samples
    print(f"Loading samples from {args.input_file}")
    samples = load_json_file(args.input_file)
    print(f"Loaded {len(samples)} samples")
    
    # Initialize judge with auto-loaded config
    judge = ReferenceValidationJudge(
        judge_model=args.judge_model,
        base_url=config["base_url"],
        api_key=config["api_key"]
    )
    
    # Validate samples
    results = judge.validate_all_samples(samples, delay_seconds=args.delay)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "reference_validation_results.json")
    save_json_file(output_file, results)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()