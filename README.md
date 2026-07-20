# AI Marketing Acceleration Platform

## Overview

**BNZ AI Marketing Acceleration Platform** is an AI-powered personalised marketing solution developed for the **University of Auckland × BNZ Hackathon (Challenge 2: AI-powered personalised marketing and accelerated delivery)**.

The platform combines **machine learning customer segmentation**, **FastAPI backend services**, and **Amazon Bedrock large language models** to automatically generate personalised banking campaigns for different behavioural customer groups.

Instead of delivering the same advertisement to every customer, the system analyses customer behaviour patterns, identifies relevant segments, and generates targeted marketing content tailored to each audience.

The solution demonstrates how machine learning and generative AI can be integrated into a banking marketing workflow while maintaining privacy-aware design principles.

---

# Key Features

- **AI-powered personalised marketing**
  - Generates segment-specific banking advertisements using Amazon Bedrock.

- **Machine learning customer segmentation**
  - Integrates with a behavioural customer segmentation service based on ML clustering.

- **End-to-end AI workflow**
  - Connects frontend, backend APIs, ML services, and cloud AI infrastructure.

- **AI customer assistant**
  - Answers customer questions using product information and generated marketing content.

- **Privacy-aware architecture**
  - Uses behavioural segments rather than personally identifiable information (PII).

---

# System Architecture

```text
                         Frontend
                    frontend/index.html
                              |
                              |
                              v
              FastAPI Marketing Backend (8010)
                              |
              +---------------+---------------+
              |                               |
              v                               v
 Bank Segmentation Service             Amazon Bedrock
        (8000)                         Nova Lite LLM
              |                               |
              |                               |
 Behavioural Customer Segments        Personalised Ad Generation
 Customer IDs                          AI Assistant Responses
              |
              |
              v

       Targeted Marketing Campaigns
```

---

# Repository Structure

This project consists of two repositories.

## 1. bank-segmentation-service

Repository:

https://github.com/Maishi226/bank-segmentation-service

This service provides:

- Machine learning customer segmentation
- Behavioural customer grouping
- Segment information API
- Customer ID retrieval API

Runs locally on:

```
http://127.0.0.1:8000
```

---

## 2. bnz-ai-marketing-hybrid

Repository:

https://github.com/Maishi226/bnz-ai-marketing-hybrid

This repository provides:

- Frontend interface
- FastAPI marketing backend
- Amazon Bedrock integration
- AI assistant workflow

Runs locally on:

```
http://127.0.0.1:8010
```

---

# Installation and Setup

## Step 1: Start Customer Segmentation Service

First, start the ML segmentation API.

```bash
cd bank-segmentation-service
bash setup_and_run.sh
```

The service should be available at:

```
http://127.0.0.1:8000
```

You can verify it through:

```
http://127.0.0.1:8000/docs
```

---

## Step 2: Configure AWS Environment

Create the local environment file:

```bash
cp .env.example .env
```

Configure your AWS profile:

```text
AWS_PROFILE=bnz-demo
AWS_REGION=ap-southeast-2
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
SEGMENTATION_SERVICE_URL=http://127.0.0.1:8000
```

AWS credentials are managed through local AWS CLI profiles or IAM Identity Center (SSO).

Do not commit `.env` files.

---

## Step 3: Start Marketing Backend

Run:

```bash
./start_backend.sh
```

The backend will start at:

```
http://127.0.0.1:8010
```

---

## Step 4: Launch Frontend

Open the frontend:

```bash
open frontend/index.html
```

---

# Demo Workflow

1. A bank employee enters a product or offer, such as:

```
Car Loan
```

2. The FastAPI backend receives the request.

3. The backend calls the customer segmentation service to retrieve:

- Behavioural customer segments
- Relevant customer IDs

4. The selected customer segments and product information are sent to Amazon Bedrock.

5. Bedrock generates personalised advertisements for each customer segment.

6. The frontend displays:

- Generated advertisements
- Target customer IDs
- AI assistant responses

---

# API Architecture

Frontend request:

```
frontend/index.html
        |
        v
http://127.0.0.1:8010/api/generate
```

Backend workflow:

```
FastAPI Backend
        |
        |
        +--> bank-segmentation-service

             GET /v1/segments

             GET /v1/model/features

             GET /v1/customers?segment_id={id}


        |
        |
        +--> Amazon Bedrock

             Generate personalised advertisements

             Generate AI assistant responses
```

---

# Machine Learning Component

The customer segmentation service applies an unsupervised learning approach:

- Behavioural banking data preprocessing
- Feature transformation and standardisation
- K-Means clustering
- Cluster evaluation using Silhouette Score
- Customer assignment into behavioural segments

These segments allow the marketing engine to generate more relevant advertisements instead of using a single generic campaign.

---

# Technologies

- Python
- FastAPI
- Amazon Bedrock (Nova Lite)
- AWS IAM Identity Center (SSO)
- Machine Learning
- K-Means Clustering
- REST APIs
- JavaScript
- HTML
- Git

---

# Privacy and Safety

The system follows privacy-aware design principles:

- Customer segmentation is based on behavioural patterns, not identity information.
- No names, passports, addresses, or uploaded identity documents are processed.
- AWS credentials are not stored in this repository.
- Local environment files, virtual environments, and caches are excluded from Git.

---

# Future Improvements

Potential future extensions:

- Deploy backend services using AWS Lambda or ECS.
- Integrate real-time customer interaction events.
- Improve segmentation using more advanced ML models.
- Add campaign performance feedback loops for continuous optimisation.
