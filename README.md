# Retail Insights Assistant

GenAI-powered MVP for retail sales summarization and conversational analytics. The app uses Streamlit, LangGraph, Pandas, DuckDB, and Vertex AI Gemini through Google Application Default Credentials.

## Features

- Upload CSV, Excel, JSON, or text report.
- Professional Streamlit dashboard with KPI, summary, chat, and schema tabs.
- Summarization mode for executive sales insights.
- Conversational Q&A mode for ad-hoc retail analytics.
- KPI snapshot and automatic insights from detected schema.
- LangGraph multi-agent pipeline:
  - Language-to-query resolution agent
  - Data extraction agent
  - Validation and answer agent
- Automatic schema detection for varied retail/order datasets such as Amazon exports.
- Read-only SQL validation before DuckDB execution.
- Architecture notes and PowerPoint slide generator for 100GB+ scale.

## Architecture

```text
User / Analyst
    |
    v
Streamlit UI on Cloud Run
    |
    |-- Upload CSV / Excel / JSON / text report
    |
    v
Data Layer
    |
    |-- Pandas loads uploaded files
    |-- Schema detector maps dataset columns to business roles
    |-- Helper columns are created:
    |      __ri_date
    |      __ri_revenue
    |      __ri_quantity
    |-- DuckDB executes analytical SQL
    |
    v
Insight Layer
    |
    |-- KPI cards
    |-- Auto insights
    |-- Summary aggregates
    |
    v
LangGraph Agent Layer
    |
    |-- Summary graph:
    |      schema_detection_node
    |      kpi_insight_node
    |      summary_generation_node
    |      summary_validation_node
    |
    |-- Q&A graph:
           schema_detection_node
           language_to_query_node
           sql_validation_node
           data_extraction_node
           validation_answer_node
    |
    v
Vertex AI Gemini
    |
    v
Final business summary or conversational answer
```

## End-to-End Flow

1. User uploads a dataset.
2. The app loads the data into Pandas.
3. The schema detector identifies roles such as date, revenue, quantity, category, SKU, order status, city, state, fulfillment, courier status, and channel.
4. KPI cards are computed locally, so the dashboard works even before calling the LLM.
5. In Summarization mode, LangGraph runs schema detection, KPI insight generation, summary generation, and summary validation nodes.
6. In Conversational Q&A mode, LangGraph runs schema detection, language-to-query, SQL validation, data extraction, and answer validation nodes.
7. The language-to-query node converts the user question to DuckDB SQL.
8. The SQL validation node ensures the SQL is read-only before execution.
9. The data extraction node runs the query in DuckDB.
10. The validation node converts the query result into a concise answer and recommendation.
11. The UI shows the final answer plus an agent trace for demo evidence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=us-central1
export VERTEX_MODEL=gemini-2.5-flash
```

## Run

```bash
streamlit run streamlit_app.py
```

Use `sample_sales.csv` or an Amazon sales CSV for a quick test. In the sidebar, confirm your Vertex project, location, and model. The default model is `gemini-2.5-flash`.

## Cloud Run Deployment Plan

This section is a deployment plan only. Do not run these commands until you are ready to deploy.

### 1. Required Google Cloud APIs

Enable these APIs in the target project:

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 2. Service Account

Create or use a Cloud Run service account with these minimum permissions:

```text
Vertex AI User
Logs Writer
```

Example:

```bash
gcloud iam service-accounts create retail-insights-runner \
  --display-name="Retail Insights Cloud Run Service Account"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:retail-insights-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:retail-insights-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter"
```

### 3. Container Requirements

Cloud Run needs a container. Add a `Dockerfile` before deploying:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV GOOGLE_CLOUD_LOCATION=us-central1
ENV VERTEX_MODEL=gemini-2.5-flash

CMD streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
```

Optional `.dockerignore`:

```text
.venv
__pycache__
*.pyc
.DS_Store
architecture.pptx
```

### 4. Build Image

```bash
gcloud builds submit \
  --tag gcr.io/YOUR_PROJECT_ID/retail-insights-assistant
```

### 5. Deploy to Cloud Run

```bash
gcloud run deploy retail-insights-assistant \
  --image gcr.io/YOUR_PROJECT_ID/retail-insights-assistant \
  --platform managed \
  --region us-central1 \
  --service-account retail-insights-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1,VERTEX_MODEL=gemini-2.5-flash \
  --memory 2Gi \
  --cpu 1 \
  --timeout 900 \
  --allow-unauthenticated
```

For private access, remove `--allow-unauthenticated` and configure IAM-based access.

### 6. Post-Deployment Test Plan

After deployment:

1. Open the Cloud Run URL.
2. Upload a small Amazon sales CSV.
3. Confirm **Detected schema** maps columns such as `Date`, `Amount`, `Qty`, `Category`, `Status`, `ship-state`, and `Courier Status`.
4. Confirm KPI cards appear.
5. Generate a summary.
6. Ask Q&A examples:

```text
Which category generated the highest revenue?
How many orders were cancelled?
Which state generated the highest revenue?
Which fulfillment method has the most shipped orders?
```

7. Capture screenshots of KPI cards, summary, Q&A, and agent trace.

## CLI Smoke Test

The smoke test uses a deterministic dummy LLM, so it does not require cloud credentials.

```bash
python test_runner.py
```

## Generate Architecture Slides

```bash
python generate_slides.py
```

This creates `architecture.pptx`.

## 100GB+ Scale Design

For production-scale data, land raw files in a cloud data lake, preprocess with Dataflow, Dataproc, Spark, or BigQuery, and store curated data as partitioned Parquet/Delta tables plus BigQuery analytical tables. The assistant should route generated SQL to BigQuery, use materialized aggregate tables for common metrics, and use vector search for unstructured reports. See [docs/architecture.md](docs/architecture.md).

## Production Scale Plan

The MVP is suitable for local files and small-to-medium CSVs. For 100GB+ datasets:

- Store raw files in a cloud data lake.
- Convert CSV to Parquet partitioned by date, region/state, and category.
- Use BigQuery or Dataproc/Spark for preprocessing and large analytical queries.
- Replace local DuckDB execution with BigQuery SQL execution.
- Keep DuckDB for uploaded small files and local demo mode.
- Store common aggregate tables for daily/monthly revenue, category performance, cancellations, fulfillment performance, and geography performance.
- Use a vector database such as FAISS, Chroma, Vertex AI Vector Search, or Pinecone for unstructured reports and business documents.
- Cache common LLM prompts, generated SQL, and query results to reduce latency and cost.
- Log generated SQL, row counts, latency, model, token usage, and validation outcomes.

## Assumptions and Limits

- The MVP detects common roles such as date, revenue/amount, quantity, category, product/SKU, status, region/location, channel, fulfillment, and size.
- SQL generation is constrained to read-only `SELECT` and `WITH` statements.
- For very large files, use a warehouse such as BigQuery instead of uploading directly to Streamlit.
- Human-reviewed query-answer examples are recommended before production use.

## Deliverables Checklist

- Code implementation: Streamlit app, agents, data layer, LLM client.
- Architecture slides: run `python generate_slides.py` to create `architecture.pptx`.
- README / technical notes: this file.
- Demo screenshots:
  - Dataset upload and data preview
  - Detected schema
  - KPI cards and auto insights
  - Summary output
  - Conversational Q&A output
  - Agent trace
