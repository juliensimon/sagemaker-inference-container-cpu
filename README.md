# An Amazon SageMaker Container for Hugging Face Inference on AWS Graviton

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Amazon SageMaker](https://img.shields.io/badge/Amazon%20SageMaker-FF9900?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![ARM64](https://img.shields.io/badge/ARM64-Graviton-orange?style=flat&logo=arm&logoColor=white)](https://aws.amazon.com/ec2/graviton/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yellow?style=flat)](https://huggingface.co/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-enabled-brightgreen?style=flat)](https://github.com/ggerganov/llama.cpp)

> **💡 Running Locally?** This container works great outside of SageMaker too! Check out [README-run-locally.md](README-run-locally.md) for Docker commands, [README-docker-compose.md](README-docker-compose.md) for Docker Compose setup, or [helm/README.md](helm/README.md) for Kubernetes deployment with Helm.

## Why?

Because small language models and modern CPUs are a great match for cost-efficient AI inference. More context in these blog posts: ["The case for small language model inference on Arm CPUs"](https://www.arcee.ai/blog/the-case-for-small-language-model-inference-on-arm-cpus) and ["Is running language models on CPU really viable?"](https://www.arcee.ai/blog/is-running-language-models-on-cpu-really-viable).

Because I've been trying for a while to collaborate with AWS and Arm on this project, and I got tired of waiting 😴

So there. Enjoy!

Caveat: I've only tested sub-10B models so far. Timeouts could hit on larger models. Bug reports, ideas, and pull requests are welcome.

## What It Does

- Based on a clean source build of llama.cpp
- Native integration with the SageMaker SDK and the Graviton3/Graviton4 instances
- Model deployment from the Hugging Face hub or an Amazon S3 bucket
- Deployment of existing GGUF models
- Deployment of safetensors models, with automatic GGUF conversion and quantization
- Support for  OpenAI API (`/v1/chat/completions`, `/v1/completions`)
- Support for streaming and non-streaming text generation
- Support for all `llama-server` flags


## Architecture

```
SageMaker Endpoint → FastAPI Adapter (port 8080) → llama.cpp Server (port 8081)
```

## Build and Push to Amazon ECR

### Prerequisites

- Docker with ARM64/multi-platform support (building on a Mac works great)
- AWS CLI configured with appropriate permissions
- ECR repository created

### 1. Build the Container

```bash
# Clone repository
git clone https://github.com/juliensimon/sagemaker-inference-container-graviton

cd sagemaker-inference-container-graviton

# Build for ARM64 (Graviton)
docker build --platform linux/arm64 -t sagemaker-inference-container-graviton .
```

### 2. Push to ECR

```bash
# Set variables
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
ECR_REPOSITORY="sagemaker-inference-container-graviton"

# Create ECR repository (if it doesn't exist)
aws ecr create-repository \
    --repository-name $ECR_REPOSITORY \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag image
docker tag sagemaker-inference-container-graviton:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
```

## Deploy to SageMaker

Here's a quick overview of how to deploy models. A full notebook is available in `examples/`.

```python

# Option 1: Deploy a safetensors model from HuggingFace Hub (auto-convert + quantize)
model_environment = {
    "HF_MODEL_ID": "your-model-repository",
    "QUANTIZATION": "Q8_0",
    "HF_TOKEN": hf_token, # optional, only for private and gated models
    "LLAMA_CPP_ARGS": llama_cpp_args # optional, see llama-server -h
}

# Option 2: Deploy a GGUF model from HuggingFace Hub
model_environment = {
    "HF_MODEL_ID": "your-model-repository-GGUF",
    "MODEL_FILENAME": "your-model.gguf"
}

# Option 3: Deploy a safetensors model from S3 (auto-convert + quantize)
model_environment = {
    "HF_MODEL_URI": "s3://your-bucket/your-model/",
    "QUANTIZATION": "Q4_0"
}

# Option 4: Deploy a GGUF model from S3
model_environment = {
    "HF_MODEL_URI": "s3://your-bucket/",
    "MODEL_FILENAME": "your-model.gguf"
}

# Create deployable model
model = Model(
    image_uri=your_image_uri,
    role=role,
    env=model_environment,
)

# Deploy the model
response = model.deploy(...)
```

## Usage

### Test the endpoint

```python
model_sample_input = {
    "messages": [
        {"role": "system", "content": "You are a friendly and helpful AI assistant."},
        {
            "role": "user",
            "content": "Suggest 5 names for a new neighborhood pet food store. Names should be short, fun, easy to remember, and respectful of pets. \
        Explain why customers would like them.",
        },
    ],
    "max_tokens": 1024
}

response = runtime_sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(model_sample_input),
)

output = json.loads(response["Body"].read().decode("utf8"))
```

### Environment Variables

| Variable | Description | Usage |
|----------|-------------|---------|
| `HF_MODEL_ID` | Hugging Face model repository  | Hub deployments |
| `HF_MODEL_URI` | S3 URI for model files (safetensors or GGUF) | S3 deployments|
| `MODEL_FILENAME` | Specific GGUF file to use | GGUF model deployment |
| `HF_TOKEN` | Hugging Face token for private and gated models | Private and gated hub models |
| `QUANTIZATION` | Quantization level (e.g., Q4_K_M) | default is F16 |
| `LLAMA_CPP_ARGS` | Additional llama.cpp arguments | default is empty |

## License

Modified MIT License
