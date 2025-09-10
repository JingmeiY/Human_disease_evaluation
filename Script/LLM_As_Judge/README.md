# LLM-as-Judge Evaluation Scripts

This directory contains scripts to perform automated evaluation of disease outbreak severity assessments using Large Language Models as judges. The evaluation replicates the human expert evaluation tasks from the Google Form but uses LLMs for faster processing.

## Overview

The evaluation consists of two main tasks:

1. **Reference Validation**: Validate the quality of GPT-o1 generated "gold standard" reference assessments
2. **Model Comparison**: Compare outputs from fine-tuned BEACON LLM vs base model

## Files

- `reference_validation_judge.py`: Script for validating reference assessments
- `model_comparison_judge.py`: Script for comparing model outputs  
- `run_llm_judge.py`: Convenient runner script for both tasks
- `config.py`: Configuration for different judge models
- `README.md`: This documentation

## Setup

### 1. Install Dependencies

```bash
pip install openai
```

### 2. Set API Keys

Set environment variables for the judge models you want to use:

```bash
# For DeepSeek (recommended)
export DEEPSEEK_API_KEY="your_deepseek_api_key"

# For Google Gemini
export GOOGLE_API_KEY="your_google_api_key"

# For OpenAI
export OPENAI_API_KEY="your_openai_api_key"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Usage

### Quick Start (Recommended)

Use the runner script to perform both tasks:

```bash
python run_llm_judge.py --input_file path/to/your/samples.json --judge deepseek
```

### Available Judge Models

List available judge models:

```bash
python run_llm_judge.py --list_judges
```

Available options:
- `deepseek`: DeepSeek Chat model (recommended, cost-effective)
- `gemini`: Google Gemini Pro (comprehensive analysis)
- `claude`: Claude 3 Sonnet (detailed reasoning)
- `openai`: GPT-4 Turbo (balanced evaluation)

### Individual Scripts

Run reference validation only:

```bash
python reference_validation_judge.py \
    --input_file samples.json \
    --output_file validation_results.json \
    --judge_model deepseek-chat
```

Run model comparison only:

```bash
python model_comparison_judge.py \
    --input_file samples.json \
    --output_file comparison_results.json \
    --judge_model deepseek-chat
```

### Advanced Options

Specify custom output directory:

```bash
python run_llm_judge.py \
    --input_file samples.json \
    --judge deepseek \
    --output_dir /path/to/results \
    --task both
```

Adjust API call delay (for rate limiting):

```bash
python run_llm_judge.py \
    --input_file samples.json \
    --judge deepseek \
    --delay 2.0
```

Run only specific task:

```bash
python run_llm_judge.py \
    --input_file samples.json \
    --judge deepseek \
    --task validation  # or 'comparison'
```

## Input Data Format

Your input JSON file should contain samples with the following structure:

```json
[
  {
    "sample_id": "23",
    "domain": "Geographic", 
    "factors_description": "Evaluates factors affecting disease transmission...",
    "article_content": "Northwestern students and staff have shown concern...",
    "reference_score": 5,
    "reference_reasoning": "All three factors scored at the highest level...",
    "model_a_score": 5,
    "model_a_reasoning": "The combination of high population density...",
    "model_b_score": 5,
    "model_b_reasoning": "All three factors score high...",
    "model_a_actual": "meta-llama",
    "model_b_actual": "Paschalidis-NOC-Lab"
  }
]
```

## Output Format

### Reference Validation Results

```json
{
  "metadata": {
    "task": "reference_validation",
    "judge_model": "deepseek-chat",
    "total_samples": 30,
    "timestamp": "2024-01-15T10:30:00"
  },
  "results": [
    {
      "sample_id": "23",
      "score_assessment": "Appropriate",
      "score_justification": "The severity score correctly reflects...",
      "reasoning_quality": "Good",
      "reasoning_justification": "The reasoning is comprehensive...",
      "overall_comments": "Strong assessment overall..."
    }
  ]
}
```

### Model Comparison Results

```json
{
  "metadata": {
    "task": "model_comparison", 
    "judge_model": "deepseek-chat",
    "total_samples": 30
  },
  "analysis": {
    "total_samples": 30,
    "valid_evaluations": 30,
    "overall_winners": {
      "model_a": 12,
      "model_b": 15,
      "ties": 3
    },
    "beacon_vs_base_analysis": {
      "beacon_wins": 18,
      "base_wins": 10,
      "ties": 2,
      "beacon_win_rate": 0.60
    }
  },
  "results": [
    {
      "sample_id": "23",
      "score_comparison": "Both equivalent",
      "reasoning_comparison": "Model B slightly better",
      "overall_winner": "Model B",
      "overall_justification": "While scores are equivalent..."
    }
  ]
}
```

## Evaluation Criteria

The LLM judges evaluate based on:

- **Scientific Accuracy**: Correctness of epidemiological and medical information
- **Risk Appropriateness**: How well the severity score matches the threat level  
- **Reasoning Quality**: Completeness and quality of justification

## Best Practices

1. **Multiple Judges**: Run evaluation with different judge models for cross-validation
2. **Rate Limiting**: Use appropriate delay between API calls to avoid rate limits
3. **Error Handling**: Check output for any failed evaluations (marked as "Error")
4. **Cost Management**: DeepSeek is recommended for cost-effective evaluation

## Troubleshooting

### API Key Issues
- Ensure environment variables are set correctly
- Check API key validity and account credits

### Rate Limiting
- Increase `--delay` parameter
- Use smaller batch sizes if needed

### JSON Parsing Errors
- Check the `raw_response` field in error cases
- Some models may need prompt adjustments

## Cost Estimation

Approximate costs per 100 samples (as of 2024):
- DeepSeek: ~$0.50-1.00
- OpenAI GPT-4: ~$5.00-10.00  
- Google Gemini: ~$2.00-4.00
- Claude 3: ~$3.00-6.00


