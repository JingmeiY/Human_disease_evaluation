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


class ReferenceValidation(BaseModel):
    """Pydantic model for structured evaluation output"""
    Q1_score_appropriateness: str  # "Too low" | "Appropriate" | "Too high"
    Q1_explanation: str
    Q2_reasoning_quality: str      # "Poor" | "Average" | "Good"
    Q2_explanation: str


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
## EVALUATION CONTEXT

**Domain:** {sample.get('domain', 'Unknown')}
**Risk Factors:** {sample.get('factors_description', 'Not specified')}

## ARTICLE CONTENT
{sample.get('article_content', 'Article content not available')}

## ASSESSMENT TO EVALUATE
**Severity Score:** {sample.get('reference_score', 'N/A')}/5
**Reasoning:** {sample.get('reference_reasoning', 'No reasoning provided')}

## SEVERITY SCALE REFERENCE
1 = Very Low Risk
2 = Low Risk  
3 = Moderate Risk
4 = High Risk
5 = Very High Risk

## EVALUATION GUIDELINES

**Scientific Accuracy:** Evaluate the correctness of epidemiological and medical information presented in the reasoning.

**Risk Appropriateness:** Assess how well the severity score matches the described threat level given the article content and domain-specific risk factors.

**Reasoning Quality:** Judge the completeness, clarity, and logical coherence of the justification provided.

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

Provide your evaluation in the following JSON format:

```json
{{
    "Q1_score_appropriateness": "<A. Too low|B. Appropriate|C. Too high>",
    "Q1_explanation": "<Brief explanation for your choice in Q1>",
    "Q2_reasoning_quality": "<A. Poor|B. Average|C. Good>", 
    "Q2_explanation": "<Brief explanation for your choice in Q2>",
}}
```
"""
        return prompt
    
    def validate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single sample and return structured output"""
        
        sample_id = sample.get('sample_id', 'unknown')
        print(f"Validating sample {sample_id}...")
        
        prompt = self.create_validation_prompt(sample)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert epidemiologist providing objective evaluations of AI-generated disease outbreak assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format=ReferenceValidation,
            )
            
            # Save raw response content directly
            raw_response = response.choices[0].message.content
            
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


class ReferenceValidation(BaseModel):
    """Pydantic model for structured evaluation output"""
    Q1_score_appropriateness: str  # "Too low" | "Appropriate" | "Too high"
    Q1_explanation: str
    Q2_reasoning_quality: str      # "Poor" | "Average" | "Good"
    Q2_explanation: str


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
## EVALUATION CONTEXT

**Domain:** {sample.get('domain', 'Unknown')}
**Risk Factors:** {sample.get('factors_description', 'Not specified')}

## ARTICLE CONTENT
{sample.get('article_content', 'Article content not available')}

## ASSESSMENT TO EVALUATE
**Severity Score:** {sample.get('reference_score', 'N/A')}/5
**Reasoning:** {sample.get('reference_reasoning', 'No reasoning provided')}

## SEVERITY SCALE REFERENCE
1 = Very Low Risk
2 = Low Risk  
3 = Moderate Risk
4 = High Risk
5 = Very High Risk

## EVALUATION GUIDELINES

**Scientific Accuracy:** Evaluate the correctness of epidemiological and medical information presented in the reasoning.

**Risk Appropriateness:** Assess how well the severity score matches the described threat level given the article content and domain-specific risk factors.

**Reasoning Quality:** Judge the completeness, clarity, and logical coherence of the justification provided.

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

Provide your evaluation in the following JSON format:

```json
{{"Q1_score_appropriateness": "<A. Too low|B. Appropriate|C. Too high>", "Q1_explanation": "<Brief explanation for your choice in Q1>", "Q2_reasoning_quality": "<A. Poor|B. Average|C. Good>", "Q2_explanation": "<Brief explanation for your choice in Q2>"}}
```
"""
        return prompt
    
    def validate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single sample and return structured output"""
        
        sample_id = sample.get('sample_id', 'unknown')
        print(f"Validating sample {sample_id}...")
        
        prompt = self.create_validation_prompt(sample)
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert epidemiologist and public health specialist conducting systematic evaluations of AI-generated disease outbreak severity assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            
            # Save raw response content directly
            raw_response = response.choices[0].message.content
            print(f"Raw response: {raw_response}")
            
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
    parser.add_argument("--judge_model", default="deepseek-reasoner", help="Judge model name")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--input_file", default="./Result/evaluation_data/selected_evaluation_samples.json", help="Path to input JSON file with samples")
    parser.add_argument("--output_dir", default="./Result/LLM_Judge/deepseek-reasoner", help="Path to input JSON file with samples")
    parser.add_argument("--judge_config_path", default="./llm_judge_config.json", help="Path to judge config file")

    args = parser.parse_args()

    # Load model configuration automatically
    config = load_json_file(args.judge_config_path)[args.judge_model]
    print(f"Using model: {args.judge_model}")
    print(f"Base URL: {config['base_url'] or 'default'}")
    print(f"API Key: {'configured' if config['api_key'] else 'missing'}")
    
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