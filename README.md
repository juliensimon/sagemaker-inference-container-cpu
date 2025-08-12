# An Amazon SageMaker Container for Hugging Face model inference on AWS Graviton instances

## What It Does

- Native integration with the SageMaker SDK
- Deployment from the Hugging Face hub or an Amazon S3 bucket
- Support for existing GGUF models
- Support for safetensors models, with automatic GGUF conversion and quantization
- Support for the OpenAI API endpoints (`/v1/chat/completions`, `/v1/completions`)
- Support for streaming and non-streaming text generation
- Support for all `llama-server` flags

## Architecture

```
SageMaker Endpoint → FastAPI Adapter (port 8080) → llama.cpp Server (port 8081)
```

## Build and Push to ECR

### Prerequisites

- Docker with ARM64/multi-platform support (building on a Mac works great)
- AWS CLI configured with appropriate permissions
- ECR repository created

### 1. Build the Container

```bash
# Clone repository
git clone https://github.com/juliensimon/sagemaker-inference-llamacpp-graviton
cd sagemaker-inference-llamacpp-graviton

# Build for ARM64 (Graviton)
docker build --platform linux/arm64 -t sagemaker-llamacpp-graviton .
```

### 2. Push to Amazon ECR

```bash
# Set variables
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
ECR_REPOSITORY="sagemaker-llamacpp-graviton"

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
docker tag sagemaker-llamacpp-graviton:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
```

## Deploy to SageMaker

Here's a quick look at how you can deploy models. A full notebook is available in `examples/`.

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
    "HF_MODEL_URI": "s3://your-bucket/your-model.gguf",
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

| Variable | Description | Mandatory |
|----------|-------------|---------|
| `HF_MODEL_ID` | Hugging Face model repository  | Only for hub deployments |
| `HF_MODEL_URI` | S3 URI for model files (safetensors or GGUF) | Only for S3 deployments|
| `MODEL_FILENAME` | Specific GGUF file to use | Only for GGUF hub deployments |
| `HF_TOKEN` | Hugging Face token for private and gated models | Only for private and gated hub models |
| `QUANTIZATION` | Quantization level (e.g., Q4_K_M) | No, F16 will be used by default |
| `LLAMA_CPP_ARGS` | Additional llama.cpp arguments | No |

## License

MIT License
