# Retail Insights Assistant

Retail Insights Assistant is a Streamlit-based analytics demo designed to turn retail transaction data into business summaries, KPI insights, and conversational answers.

## What the application does

The application enables a user to provide retail sales data and then:

- automatically detects the dataset schema and maps retail fields such as order date, revenue, quantity, category, location, fulfillment status, and courier status
- computes core business KPIs for revenue, volume, average order value, cancellations, and top categories
- generates an executive summary describing trends and anomalies
- supports conversational questions about the dataset and answers them with validated analytics
- preserves a clean, read-only query pipeline so generated queries are not allowed to modify data

## High-level architecture

The project is organized in three main layers:

1. User interface
   - A Streamlit application provides the dashboard, chart views, summary card, and conversational prompt area.
   - The UI is responsible for data upload selection, configuration values, and presenting results.

2. Data layer
   - The data layer loads retail files, normalizes schema elements, and prepares the dataset for query execution.
   - It detects key columns and builds helper fields for date, revenue, quantity, and category mapping.
   - DuckDB is used as the analytics engine for fast, in-memory SQL queries over the loaded dataset.

3. Intelligence layer
   - The intelligence layer uses a language model to generate summary text and to translate user questions into SQL queries.
   - Query generation is validated to ensure only read-only analytics queries are executed.
   - Results are converted back into natural language answers and suggestions.

## Application flow

The user-visible workflow proceeds as follows:

- Data input: the user supplies a dataset.
- Schema detection: the app identifies retail fields and labels them for analytics.
- KPI calculation: core metrics are computed from the normalized dataset.
- Summary generation: the system creates a business overview from the detected schema and KPI results.
- Conversational analytics: a user question is translated into a SQL query, validated, executed, and answered.

## Key components

- `streamlit_app.py`: main application entrypoint and UI orchestration.
- `data_layer.py`: dataset loading, schema normalization, and DuckDB preparation.
- `llm_client.py`: language model client abstraction.
- `agents.py` and `agent_graph.py`: orchestration of multi-step language and analytics processing.
- `deploy.sh` and `Dockerfile`: container deployment support.
- `test_runner.py`: repository smoke test harness.

## Deployment model

The expected deployment model is:

- containerized Streamlit service running on a managed platform such as Cloud Run
- Google Cloud Storage or a shared data bucket as the source for retail input files
- Vertex AI or a comparable language model service for summary and question answering
- a service account with permissions to read data and call model APIs

## Design goals

This project is designed to:

- make retail analytics approachable for business users
- validate all generated queries before execution
- keep analytics read-only and transparent
- separate UI, data preparation, and intelligence logic
- allow the application to work with real retail data and cloud-hosted model services

## Who should read this repository

This README is intended for developers, product reviewers, and evaluators who want to understand:

- the application purpose
- the architecture and component responsibilities
- the user workflow from dataset to insight
- the deployment pattern without needing to execute any commands

## Notes

- The repository is not limited to one dataset format; it is built to support retail-style CSVs and structured export files.
- The application emphasizes schema detection, KPI extraction, and conversational analytics rather than raw data transformation.
- The architecture is intentionally lightweight so the app can be deployed quickly while preserving an extensible analytics pipeline.
