---
layout: default
title: "DATA 202 Module 9: Model Deployment and MLOps"
---

# DATA 202 Module 9: Model Deployment and MLOps

## Introduction

A model in a Jupyter notebook helps no one. The value of machine learning comes when models make predictions in production—powering applications, informing decisions, serving users. This transition from prototype to production is where many data science projects fail.

This module covers the engineering practices needed to deploy and maintain ML systems: containerization, serving infrastructure, monitoring, and the emerging discipline of MLOps.

---

## Part 1: From Notebook to Production

### The Deployment Gap

Why do so many ML projects never reach production?

**Technical Challenges**:
- Dependency management
- Environment differences (dev vs. prod)
- Scalability requirements
- Latency constraints
- Model versioning

**Organizational Challenges**:
- Handoff between data science and engineering
- Unclear ownership
- Lack of monitoring
- No feedback loop

### The ML Lifecycle

1. **Development**: Experimentation, model training
2. **Validation**: Testing, review, approval
3. **Deployment**: Serving in production
4. **Monitoring**: Performance tracking, drift detection
5. **Retraining**: Model updates based on new data

MLOps applies DevOps principles to this lifecycle.

---

## Part 2: Packaging and Containerization

### Environment Management

Reproducible environments are essential:

**requirements.txt**:
```
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

**Conda environment.yml**:
```yaml
name: ml-app
dependencies:
  - python=3.10
  - scikit-learn=1.3.0
  - pandas=2.0.3
```

### Docker for ML

**Docker** packages code with dependencies in isolated containers.

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run**:
```bash
docker build -t ml-model .
docker run -p 8000:8000 ml-model
```

---

## Part 3: Model Serving

### Serving Approaches

**Batch Inference**: Process large datasets offline
- Daily predictions
- Stored in database
- Simple infrastructure

**Real-Time Inference**: On-demand predictions
- API endpoints
- Low latency requirements
- Scalability challenges

**Edge Inference**: Run on device
- Mobile apps
- IoT devices
- Privacy preservation

### Building APIs with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

class PredictionOutput(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionOutput)
def predict(input: PredictionInput):
    features = [[input.feature1, input.feature2, input.feature3]]
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0].max()
    return PredictionOutput(prediction=pred, probability=prob)
```

### Scaling Inference

**Horizontal scaling**: Multiple instances behind load balancer
**Model optimization**: Quantization, pruning, distillation
**Caching**: Cache repeated predictions
**Batching**: Process multiple requests together

### Serving Platforms

- **TensorFlow Serving**: Google's solution for TF models
- **TorchServe**: PyTorch model serving
- **Triton Inference Server**: NVIDIA, multiple frameworks
- **Seldon Core**: Kubernetes-native
- **BentoML**: Framework-agnostic, easy deployment

---

## Part 4: Monitoring and Maintenance

### What to Monitor

**System Metrics**:
- Latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates
- Resource usage (CPU, memory, GPU)

**Model Metrics**:
- Prediction distributions
- Feature distributions
- Actual outcomes (when available)
- Accuracy over time

### Data and Model Drift

**Data Drift**: Input distribution changes
- New user demographics
- Seasonal patterns
- Market shifts

**Concept Drift**: Relationship between input and output changes
- User preferences evolve
- What "spam" means changes

**Detection**:
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("drift_report.html")
```

### When to Retrain

Triggers for retraining:
- Scheduled (weekly, monthly)
- Performance degradation
- Significant drift detected
- New data available

---

## Part 5: MLOps Best Practices

### Version Control

Track not just code but:
- **Data**: DVC, Delta Lake
- **Models**: MLflow, Weights & Biases
- **Experiments**: MLflow, Neptune
- **Pipelines**: Kubeflow, Airflow

### CI/CD for ML

**Continuous Integration**:
- Automated testing
- Data validation
- Model validation

**Continuous Deployment**:
- Automated model deployment
- Canary releases
- A/B testing

### Feature Stores

Centralized feature management:
- Consistent features between training and serving
- Feature reuse across models
- Point-in-time correct features

**Platforms**: Feast, Tecton, Hopsworks

---

## DEEP DIVE: When ML Systems Fail in Production

### The Knight Capital Disaster

On August 1, 2012, Knight Capital Group lost $440 million in 45 minutes due to a software deployment failure. While not specifically ML, it illustrates deployment risks.

A technician forgot to update a server with new code. Old code interpreted new trading signals incorrectly. Automated systems amplified the error, buying high and selling low at massive scale.

### ML-Specific Failures

**Tay Chatbot (Microsoft, 2016)**: Learned from user interactions. Within hours, trolls trained it to produce offensive content. Shut down within 16 hours.

**Amazon Hiring Tool**: Trained on historical hiring data. Learned to penalize women because past hires were mostly men. Abandoned after attempted fixes failed.

**Healthcare Algorithm Bias**: The Optum algorithm (discussed in ethics module) showed racial bias discovered years after deployment—only caught through external research.

### Lessons

1. **Testing isn't enough**: Production conditions differ from test
2. **Monitor continuously**: Catch problems early
3. **Plan for rollback**: Quick reversion when things go wrong
4. **Human oversight**: Critical decisions need human review
5. **Adversarial thinking**: Consider how systems can be attacked or misused

---

## HANDS-ON EXERCISE: Deploying a Model

### Part 1: Create Serving Application

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

# Load model at startup
model = joblib.load("model.pkl")

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: Features):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0].tolist()
    return {
        "prediction": int(prediction),
        "probabilities": probabilities
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
```

### Part 2: Dockerize

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Part 3: Add Monitoring

```python
import time
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUESTS = Counter('prediction_requests_total', 'Total prediction requests')
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
def predict(features: Features):
    REQUESTS.inc()
    start = time.time()

    # ... prediction code ...

    LATENCY.observe(time.time() - start)
    return result

# Start metrics server
start_http_server(9090)
```

---

## Recommended Resources

### Books
- *Designing Machine Learning Systems* by Chip Huyen
- *Building Machine Learning Pipelines* by Hapke and Nelson
- *Machine Learning Engineering* by Andriy Burkov

### Tools
- **MLflow**: Experiment tracking and model registry
- **Kubeflow**: Kubernetes ML pipelines
- **Weights & Biases**: Experiment tracking
- **Evidently**: ML monitoring
- **Great Expectations**: Data validation

### Platforms
- **AWS SageMaker**: End-to-end ML
- **Google Vertex AI**: GCP ML platform
- **Azure ML**: Microsoft ML service
- **Databricks**: Unified analytics platform

---

*Module 9 covers model deployment and MLOps—the engineering practices that turn notebook prototypes into production systems. From containerization to monitoring to handling failures, we learn what it takes to keep ML systems running reliably in the real world.*
