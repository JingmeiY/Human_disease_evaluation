# Step 1: Prepare Test Dataset

## Running the Script
To prepare the test dataset, run the following command:
```bash
python Script/prepare_test_dataset.py \
--input_path ./Data/Merged/Severity_data/test.json \
--output_path ./Data/Processed/Severity_data/test.json
```

## Data Schema

### Input Data Schema
The input data should be in JSON format with the following structure:
```json
{
    "instruction": "Expert AI assistant prompt for disease evaluation...",
    "input": "Context and signal data about the disease outbreak...",
    "output": "JSON formatted severity assessment..."
}
```

### Output Data Schema
The script processes the input and generates output in the following format:
```json
  {
    "sample_id": 1,
    "domain": "External_risk",
    "article_id": 1,
    "article_content": "News article or signal content...",
    "instruction": "Complete evaluation instruction...",
    "input": "Disease outbreak context and details...",
    "GT": "Ground truth severity assessment in JSON format"
}
```

# Step 2: Model Deployment

This section describes how to deploy models using vLLM.

## Prerequisites

1. Docker installation with GPU support
2. NVIDIA drivers and CUDA toolkit
3. Hugging Face account and access token (for gated models)

## Model Configuration

The project supports two types of model configurations:

### 1. Meta-LLaMA Models (`meta_model_config.json`)
Uses the base Meta-LLaMA models for all evaluation tasks.

### 2. NOC Lab Fine-tuned Models (`noc_model_config.json`) 
Uses specialized fine-tuned models from the NOC Lab for different evaluation tasks:
- **NonSeverity**: `Paschalidis-NOC-Lab/Llama-3.1-8B-Full-Non-severity`
- **Severity**: `Paschalidis-NOC-Lab/Llama-3.1-8B-Full-Severity`  
- **Extraction**: `Paschalidis-NOC-Lab/Llama-3.1-8B-Full-Extraction`
- **Other factors**: Use the Severity model for Pathogen, External_risk, Clinical_impact, Transmission, Geographic, and Healthcare_system evaluations

## Environment Setup

### Set Environment Variables

For Meta models:
```bash
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_CONFIG_PATH="meta_model_config.json"
export HF_TOKEN="your_huggingface_token_here"
```

**Note**: If you want to use a different model size (e.g., Llama-3.1-70B-Instruct), you need to:
1. Update the `MODEL_NAME` environment variable
2. Manually update all model entries in `meta_model_config.json` to match your chosen model
3. The default configuration uses Llama-3.1-8B-Instruct for all evaluation tasks

For NOC Lab fine-tuned models:
```bash
export MODEL_NAME="Paschalidis-NOC-Lab/Llama-3.1-8B-Full-Severity"
export MODEL_CONFIG_PATH="noc_model_config.json"  
export HF_TOKEN="your_huggingface_token_here"
```

**Note**: If you want to use different model sizes from the NOC Lab, you need to:
1. Update the `MODEL_NAME` environment variable to your preferred model
2. Manually update the corresponding model entries in `noc_model_config.json` 
3. The default configuration uses Llama-3.1-8B variants optimized for human disease evaluation tasks
4. Available NOC Lab models include different sizes and specializations - check the [NOC Lab Hugging Face page](https://huggingface.co/Paschalidis-NOC-Lab) for available models

## Model Deployment


### Deploy Model with vLLM
```bash
# pull if not exists, otherwise use existing
docker images beacon-app | grep -q beacon-app || docker pull jingmeiyang/beacon-app && docker tag jingmeiyang/beacon-app beacon-app
```


# Start the vLLM server
```bash
docker run -d --gpus all -v ./:/app \
    -e HF_HOME="HF_HOME" \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -p 8000:8000 --name vllm-container beacon-app \
    vllm serve "$MODEL_NAME" --trust-remote-code \
    --host 0.0.0.0 --port 8000 --enable-prefix-caching
```

### Verify Deployment
```bash
# Check if the server is running
curl http://localhost:8000/v1/models

# Monitor deployment logs
docker logs vllm-container
```

## Running Evaluations

### Start Evaluation Container
Once the vLLM server is running, start an evaluation container:
```bash
docker run --rm -it --network container:vllm-container --gpus all -v ./:/app \
    -e HF_HOME="HF_HOME" -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e MODEL_CONFIG_PATH=$MODEL_CONFIG_PATH -e MODEL_NAME=$MODEL_NAME \
    --name beacon-container beacon-app bash
```

### Run Severity Evaluation
Inside the evaluation container, run the evaluation script:

```bash
python Script/evaluate_severity.py \
    --input_json ./Data/Processed/Severity_data/test.json \
    --output_json ./Result/$MODEL_NAME/Severity_data/predictions.json \
    --config $MODEL_CONFIG_PATH
```



### Useful Commands
```bash
# Check container status
docker ps -a

# View container logs
docker logs vllm-container

# Check GPU usage
nvidia-smi

# Stop all containers
docker stop $(docker ps -aq)
```


