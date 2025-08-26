# SageMaker Inference Container for Hugging Face on AWS Graviton - Helm Chart

This Helm chart deploys the SageMaker inference container for Hugging Face models on AWS Graviton instances in Kubernetes.

## Prerequisites

- Kubernetes cluster with AMD64/ARM64 nodes (assuming ARM64 in the rest of the doc)
- Helm 3.x
- kubectl configured to access your cluster
- Docker (for local image building)

## Quick Start

### 1. Build the Docker Image

First, build the Docker image for your target architecture:

```bash
# For ARM64 (Graviton) - Production
docker build --platform linux/arm64 -t sagemaker-inference-llamacpp-graviton:latest .
```

### 2. Install the Chart

```bash
# Basic installation with default values
helm install my-inference ./sagemaker-inference-llamacpp-graviton

# Or with custom values
helm install my-inference ./sagemaker-inference-llamacpp-graviton \
  --set model.hfModelId="arcee-ai/arcee-lite" \
  --set model.quantization="Q4_K_M"
```

### 3. Access the Service

```bash
# Port forward to access the service
kubectl port-forward svc/my-inference-sagemaker-inference-llamacpp-graviton 8080:8080

# Test the API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

## Values Files

The chart includes two main values files:

- **`values.yaml`** - Default production values with gated model configuration
- **`values-local.yaml`** - Local testing values with non-gated model for development

## Configuration

### Model Configuration

The chart supports different model deployment scenarios:

#### Hugging Face Hub Model (Public)
```yaml
model:
  hfModelId: "arcee-ai/arcee-lite"
  quantization: "Q4_K_M"
```

#### Hugging Face Hub Model (Gated/Private)
```yaml
model:
  hfModelId: "arcee-ai/AFM-4.5B"  # Gated model
  quantization: "Q4_K_M"
  hfToken: "your-hf-token"  # Required for gated models
```

#### S3 Model
```yaml
model:
  hfModelUri: "s3://your-bucket/your-model/"
  modelFilename: "model.gguf"
  quantization: "Q4_0"
```

#### Pre-converted GGUF Model
```yaml
model:
  hfModelId: "your-model-repository-GGUF"
  modelFilename: "your-model.gguf"
```

### Resource Configuration

```yaml
resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 2000m
    memory: 4Gi
```

### Persistence Configuration

```yaml
persistence:
  enabled: true
  storageClass: "gp3"  # EBS storage class
  accessMode: ReadWriteOnce
  size: 10Gi
```

### Service Configuration

```yaml
service:
  type: ClusterIP  # or LoadBalancer, NodePort
  port: 8080
```

## Testing

### Automated Testing

The chart includes a comprehensive test script that validates all functionality:

```bash
# Run the full test suite
cd helm
./test-helm-chart.sh
```

### Local Testing with Kind/Docker Desktop

For local testing with kind or Docker Desktop Kubernetes:

1. **Build the local image**:
   ```bash
   docker build -f Dockerfile.local -t sagemaker-inference-llamacpp-graviton:latest .
   ```

2. **Load image into kind cluster** (if using kind):
   ```bash
   kind load docker-image sagemaker-inference-llamacpp-graviton:latest --name <cluster-name>
   ```

3. **Deploy with local image**:
   ```bash
   helm install test-local ./sagemaker-inference-llamacpp-graviton \
     -f sagemaker-inference-llamacpp-graviton/values-local.yaml \
     --set image.pullPolicy=Never \
     --timeout=15m
   ```

4. **Monitor progress**:
   ```bash
   # Check pod status
   kubectl get pods -l app.kubernetes.io/instance=test-local

   # View logs with progress
   kubectl logs -l app.kubernetes.io/instance=test-local -f
   ```

5. **Test inference API**:
   ```bash
   # Wait for pod to be ready
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=test-local --timeout=600s

   # Port forward to access the service
   kubectl port-forward svc/test-local-sagemaker-inference-llamacpp-graviton 8080:8080 &

   # Test health endpoint
   curl -f http://localhost:8080/ping

   # Test chat completions
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Hello! Can you tell me a short joke?"}
       ],
       "max_tokens": 50,
       "temperature": 0.7
     }'

   # Test completions
   curl -X POST http://localhost:8080/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The capital of France is",
       "max_tokens": 20,
       "temperature": 0.7
     }'
   ```

The test script performs:
- ✅ Chart linting validation
- ✅ Template rendering validation
- ✅ Basic installation test (with nginx)
- ✅ Service creation test
- ✅ PVC creation test
- ✅ Configuration variations test

### Manual Testing

#### 1. Test with Non-Gated Model
```bash
helm install test-inference ./sagemaker-inference-llamacpp-graviton \
  -f sagemaker-inference-llamacpp-graviton/values-local.yaml \
  --set resources.limits.cpu="2000m" \
  --set resources.limits.memory="4Gi"
```

#### 2. Test with Gated Model (Requires HF Token)
```bash
helm install test-gated ./sagemaker-inference-llamacpp-graviton \
  --set model.hfModelId="arcee-ai/AFM-4.5B" \
  --set model.quantization="Q4_K_M" \
  --set model.hfToken="your-hf-token-here"
```

#### 3. Test S3 Model Deployment
```bash
helm install test-s3 ./sagemaker-inference-llamacpp-graviton \
  --set model.hfModelUri="s3://my-bucket/models/" \
  --set model.modelFilename="model-f16.Q4_K_M.gguf" \
  --set persistence.storageClass="gp3"
```

#### 4. Test LoadBalancer Service
```bash
helm install test-lb ./sagemaker-inference-llamacpp-graviton \
  -f sagemaker-inference-llamacpp-graviton/values-local.yaml \
  --set service.type=LoadBalancer
```

## Examples

### Example 1: Deploy with Public Hugging Face Model
```bash
helm install public-inference ./sagemaker-inference-llamacpp-graviton \
  -f sagemaker-inference-llamacpp-graviton/values-local.yaml \
  --set resources.limits.cpu="4000m" \
  --set resources.limits.memory="8Gi"
```

### Example 2: Deploy with Gated Model (Production)
```bash
helm install gated-inference ./sagemaker-inference-llamacpp-graviton \
  --set model.hfModelId="arcee-ai/AFM-4.5B" \
  --set model.quantization="Q4_K_M" \
  --set model.hfToken="hf_your_token_here" \
  --set persistence.storageClass="gp3" \
  --set service.type=LoadBalancer
```

### Example 3: Deploy with S3 Model
```bash
helm install s3-inference ./sagemaker-inference-llamacpp-graviton \
  --set model.hfModelUri="s3://my-bucket/models/" \
  --set model.modelFilename="model-f16.Q4_K_M.gguf" \
  --set persistence.storageClass="gp3"
```

### Example 4: Deploy with Custom Values File
```bash
# Create a custom values file
cat > my-values.yaml << EOF
model:
  hfModelId: "arcee-ai/arcee-lite"
  quantization: "Q8_0"

resources:
  limits:
    cpu: 2000m
    memory: 4Gi

service:
  type: LoadBalancer
EOF

# Deploy with custom values
helm install custom-inference ./sagemaker-inference-llamacpp-graviton -f my-values.yaml
```

## Values Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `sagemaker-inference-llamacpp-graviton` |
| `image.tag` | Container image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `model.hfModelId` | Hugging Face model ID | `arcee-ai/AFM-4.5B` |
| `model.hfModelUri` | S3 URI for model files | `""` |
| `model.modelFilename` | Specific GGUF filename | `""` |
| `model.quantization` | Quantization level | `Q4_K_M` |
| `model.hfToken` | Hugging Face token (for gated models) | `""` |
| `model.llamaCppArgs` | Additional llama.cpp arguments | `""` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.size` | Storage size | `10Gi` |
| `persistence.storageClass` | Storage class | `""` |
| `service.type` | Service type | `ClusterIP` |
| `resources.limits.cpu` | CPU limit | `4000m` |
| `resources.limits.memory` | Memory limit | `8Gi` |
| `healthCheck.enabled` | Enable health checks | `true` |

## Troubleshooting

### Common Issues

#### 1. Image Pull Errors
```bash
# Check if image exists locally
docker images sagemaker-inference-llamacpp-graviton

# For local testing, use imagePullPolicy: Never
helm install test ./sagemaker-inference-llamacpp-graviton \
  --set image.pullPolicy=Never
```

#### 2. Gated Model Access Denied
```bash
# Ensure HF_TOKEN is set for gated models
helm install test ./sagemaker-inference-llamacpp-graviton \
  --set model.hfModelId="arcee-ai/AFM-4.5B" \
  --set model.hfToken="your-hf-token"
```

## Production Deployment

### 1. Build and Push Image
```bash
# Build for ARM64 (Graviton)
docker build --platform linux/arm64 -t your-registry/sagemaker-inference-llamacpp-graviton:latest .

# Push to registry
docker push your-registry/sagemaker-inference-llamacpp-graviton:latest
```

### 2. Create Production Values
```yaml
# production-values.yaml
image:
  repository: your-registry/sagemaker-inference-llamacpp-graviton
  tag: "latest"
  pullPolicy: Always

model:
  hfModelId: "arcee-ai/AFM-4.5B"
  quantization: "Q4_K_M"
  hfToken: "your-hf-token"

persistence:
  enabled: true
  storageClass: "gp3"
  size: 20Gi

service:
  type: LoadBalancer

resources:
  limits:
    cpu: 8000m
    memory: 16Gi
  requests:
    cpu: 4000m
    memory: 8Gi
```

### 3. Deploy
```bash
helm install production-inference ./sagemaker-inference-llamacpp-graviton \
  -f production-values.yaml \
  --namespace production \
  --create-namespace
```

## Uninstalling

```bash
# Uninstall a specific release
helm uninstall my-inference

# Clean up all resources
kubectl delete pvc -l app.kubernetes.io/name=sagemaker-inference-llamacpp-graviton
```

## Contributing

This Helm chart is part of the [sagemaker-inference-container-graviton](https://github.com/juliensimon/sagemaker-inference-container-graviton) project.
