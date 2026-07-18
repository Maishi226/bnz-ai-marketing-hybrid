# BNZ AI Marketing Acceleration Hybrid Demo


This project is a local hybrid demo for Challenge 2: AI-powered personalised marketing and accelerated delivery.

It connects four parts:

1. A local frontend where a bank employee can type any product or offer.
2. A local FastAPI backend that orchestrates the workflow.
3. The existing `bank-segmentation-service` ML API, which returns behavioural customer segments and customer IDs.
4. Amazon Bedrock, which generates segment-specific ad copy and answers customer questions through an AI assistant flow.

## Required Repositories

This demo is split into two repositories:

1. [bank-segmentation-service](https://github.com/Maishi226/bank-segmentation-service)  
   Runs the ML behavioural customer segmentation API on `http://127.0.0.1:8000`.

2. [bnz-ai-marketing-hybrid](https://github.com/Maishi226/bnz-ai-marketing-hybrid)  
   Runs the frontend, hybrid FastAPI backend, Amazon Bedrock ad generation, and AI assistant.

You must start `bank-segmentation-service` first, then start this project.

## Architecture

```text
frontend/index.html
  -> http://127.0.0.1:8010/api/generate
  -> bank-segmentation-service on http://127.0.0.1:8000
       GET /v1/segments
       GET /v1/model/features
       GET /v1/customers?segment_id={id}
  -> Amazon Bedrock
  -> personalised ad copy + routed customer IDs + assistant answers
```

## Required companion service

This demo requires the separate ML segmentation repo to be running locally:

```bash
cd /path/to/bank-segmentation-service
bash setup_and_run.sh
```

The segmentation service must be available at:

```text
http://127.0.0.1:8000
```

## Environment setup

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` for your local AWS profile:

```text
AWS_PROFILE=your-local-aws-profile
AWS_REGION=ap-southeast-2
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
SEGMENTATION_SERVICE_URL=http://127.0.0.1:8000
```

Do not commit `.env`. It is intentionally ignored by Git.

## Start the hybrid backend

```bash
./start_backend.sh
```

The backend runs at:

```text
http://127.0.0.1:8010
```

## Open the frontend

```bash
open frontend/index.html
```

## Demo flow

1. Type any BNZ product or offer into the product box.
2. Click **Generate Personalised Ads with Bedrock**.
3. The backend calls the ML segmentation service to retrieve behavioural segments and customer IDs.
4. Bedrock generates different ads for each behavioural segment.
5. The UI shows which customer IDs would receive each ad.
6. The AI assistant answers customer questions using the selected product, segment, and generated ad.

## Privacy and safety notes

- The ML service returns behavioural segments, not identity labels.
- The demo does not use names, passports, addresses, or uploaded identity documents.
- `.env`, virtual environments, Python caches, and local machine settings are excluded from Git.
- AWS credentials are not stored in this repository. Use local AWS CLI profiles or SSO.
