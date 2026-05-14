# Architecture & Scaling (100GB+)

## Overview
The MVP separates data access, LLM prompting, and agent orchestration. Locally, Streamlit accepts CSV, Excel, JSON, or text reports, Pandas loads the data, DuckDB executes analytical SQL, and Vertex AI Gemini is called through Application Default Credentials. The Q&A path uses three explicit agents: language-to-query resolution, data extraction, and validation/answer generation.

## Data Engineering & Preprocessing
- Ingest CSVs into a data lake (GCS) as Parquet partitioned by date/region.
- Use batch jobs (Cloud Dataflow / Dataproc / Databricks) or PySpark to clean, deduplicate, and compute aggregates.
- Use CDC or Pub/Sub for streaming ingest when near-real-time required.

## Storage & Indexing
- Raw data: GCS (Parquet)
- Analytical warehouse: BigQuery or Snowflake for ad-hoc SQL on 100GB+ data
- Local analytical engine: DuckDB for small-to-medium datasets and prototyping
- Serving aggregates: BigQuery materialized views or partitioned aggregate tables by date, region, category, and product.

## Retrieval & Query Efficiency
- Use metadata filtering (date range, region, product) to limit scans.
- Create precomputed aggregates (daily/weekly/monthly) and summary tables.
- For natural-language retrieval, build embedding index (FAISS/Chroma/Pinecone) of reports and summaries for RAG.
- The language-to-query agent emits SQL against curated tables; the extraction agent retrieves only the resulting slice; the validation agent answers from that slice only.

## Model Orchestration
- Keep prompt templates and chain steps: intent -> retrieval -> context augmentation -> answer.
- Cache frequent query results and LLM responses.
- Use Vertex AI / OpenAI with batching and temperature control to manage cost/latency.
- Use low-temperature Gemini for SQL generation and validation, with deterministic SQL checks before execution.

## Monitoring & Evaluation
- Metrics: query latency (p95), cost per query, correctness score (human-reviewed samples), LLM confidence / hallucination rate.
- Fallbacks: when LLM output confidence low, return raw aggregates and flag for human review.
- Log generated SQL, execution time, row counts, token usage, and validation outcomes for evaluation.

## Example Query Pipeline
1. User asks: "Which region has the highest sales?"
2. Language-to-query agent creates: `SELECT region, SUM(sales) AS total_sales FROM sales GROUP BY region ORDER BY total_sales DESC LIMIT 1`.
3. Data extraction agent validates read-only SQL and runs it in DuckDB or BigQuery.
4. Validation agent converts the result into a concise business answer with a recommendation.
