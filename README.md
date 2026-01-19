# Architectural Overview of the LLM Ops Debugging and Analytics Pipeline

## Introduction

This system constitutes a unified operational analytics pipeline for Large Language Model (LLM) deployments, integrating observability, embedding-based semantic retrieval, clustering, and agentic reasoning. The system captures user interactions, models failure behavior, and enables downstream debugging through Retrieval-Augmented Generation (RAG) and specialized LLM agents. A Streamlit dashboard provides an interactive interface for developers and operators.

The system reflects emerging design principles in LLM product infrastructure, including failure-aware embeddings, semantic clustering, and agentic specialization for domain-specific diagnostics (e.g., performance vs. failure analysis).

---

## High-Level Architecture

The architecture decomposes into four conceptual layers:

```

+--------------------------------------------------+
|  User Interface (Dashboard + Agents)             |
+--------------------------------------------------+
|  Semantic & Analytical Compute (Clustering, RAG) |
+--------------------------------------------------+
|  Storage & Vector Index (PostgreSQL + pgvector)  |
+--------------------------------------------------+
|  Data Ingestion & ETL Pipeline                   |
+--------------------------------------------------+
```


```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "rgb(14,17,22)",
    "primaryColor": "#1f2937",
    "primaryTextColor": "#e5e7eb",
    "lineColor": "#9ca3af",
    "secondaryColor": "#111827",
    "fontFamily": "Inter, Arial"
  }
}}%%
flowchart TD

    %% ===== External Data Sources =====
    A1[Raw Interaction Logs]
    A2[Agent Performance Metrics]
    A3[Customer Queries & Sessions]
    A4[Failure & Escalation Outcomes]

    %% ===== Ingestion Layer =====
    B1[Log Collector]
    B2[Preprocessor]
    B3[Feature Extractor]

    %% ===== Shared Data =====
    C[(Unified Data Lake)]
    D[(Analytical Feature Store)]

    %% ===== Cluster Agent =====
    E1["Compute Embeddings
(PCA + UMAP)"]
    E2[Run KMeans / HDBSCAN]
    E3[Label Clusters]

    %% ===== Ranking Agent =====
    F1[Priority Scoring Functions]
    F2[Apply Size to Failure Ordering]
    F3[Mixed Priority Strategies]

    %% ===== Insight + Narrative Agent =====
    G1[Cluster Semantics]
    G2[Failure Interpretation]
    G3[Narrative Report Generation]

    %% ===== Dashboard / UI Agent =====
    H1[Serve Tables & Charts]
    H2[Serve UMAP / PCA Plots]
    H3[Client Interface API]

    %% ===== Outputs =====
    O1[Interactive Dashboard]
    O2[JSON Cluster Metadata]
    O3[UMAP / PCA Visualizations]
    O4[Analytical Report]

    %% ===== Connections =====
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1

    B1 --> B2 --> B3 --> C
    C --> D

    D --> E1 --> E2 --> E3
    E3 --> F1 --> F2 --> F3
    F3 --> G1 --> G2 --> G3
    G3 --> H1 --> H2 --> H3

    H3 --> O1
    G3 --> O4
    E3 --> O2
    E2 --> O3
```


Each layer communicates through well-defined interfaces and data artifacts (CSV, Parquet, embeddings, vectors).

---

## Deployed UI Dashboard
<img width="782" height="611" alt="82423eeb-4c3f-4efe-a004-ec24df4bd735" src="https://github.com/user-attachments/assets/bd9ce6fc-c5cf-4888-ac0e-da5d84cdc1f9" />

<img width="757" height="420" alt="86bacd7c-eadf-4e2c-81b5-a327e5b5d14f" src="https://github.com/user-attachments/assets/9f9f6655-d3fc-41ca-968b-bd06faab2576" />

<img width="794" height="518" alt="d6effb3f-14c8-4213-a112-d99fcdb3c4af" src="https://github.com/user-attachments/assets/6ae1c506-82f8-4885-84c8-f334013aa957" />

<img width="350" height="400" alt="0cf6cbb0-5028-4214-9a08-9619015bc8c4" src="https://github.com/user-attachments/assets/a598999b-2f3d-455e-bd14-5237538f5d2c" />

<img width="350" height="700" alt="75a67b8d-fe61-48fd-bba9-3d7eb6222a53" src="https://github.com/user-attachments/assets/f2a24efa-3395-4d63-b8bb-83f761562521" />

## Data Ingestion and ETL Pipeline

### Inputs

Data Set Used:
LLM System Ops Telemetry (Synthetic)
A synthetic, production-style, multi-table LLM telemetry dataset for LLMOps analytics and decision-grade experiments.
One row = one interaction (request → response), aggregated into sessions and users, with tool usage, failures, safety flags, latency, tokens, synthetic estimated cost, and user feedback — plus a 1:1 aligned SFT table and a prompt config dimension.

URL: https://www.kaggle.com/datasets/tarekmasryo/llm-system-ops-production-telemetry-and-sft

Dataset at a glance
Time (UTC): 2025-02-01 → 2025-04-30
Core grain: 1 row = 1 LLM interaction
Designed for: monitoring, reliability, evaluation, cost control, tool-use analytics, failure RCA
Split: split is a deterministic, group-safe train/val/test assignment derived from session_id hashing (prevents session leakage)
Row counts

llm_system_interactions.csv: 9,000
llm_system_sessions_summary.csv: 1,595
llm_system_users_summary.csv: 438
llm_system_prompts_lookup.csv: 36
llm_system_instruction_tuning_samples.csv: 9,000


The pipeline ingests four structured data sources:

- llm_system_interactions.csv
- llm_system_sessions_summary.csv
- llm_system_users_summary.csv
- llm_system_prompts_lookup.csv

These correspond to canonical LLM observability signals:

Dimension | Purpose
--------- | -------
Session | Group requests
User | Attribution information
Prompt | Request semantics
Interaction | Full text + performance + failure flags

---

### Transformations

The ETL process performs:

- Column harmonization  
  - Standardizes request_text → prompt_text

- Join operations  
  - Resolves relational context across tables

- Feature engineering  
  - Latency-per-token metrics  
  - Failure classification flags  
  - Cost and tokenization normalization  

- Failure slice extraction  
  - Separates failure population for downstream statistics

- RAG embedding computation  
  - Uses Gemini embeddings (text-embedding-004)  
  - Produces 768-dimensional semantic embeddings  

- Artifact persistence  
  - Writes:
    - etl_output.csv (operational metrics)
    - rag_interactions.parquet (semantic embeddings + metadata)

These outputs form the canonical data foundation for later analysis stages.

---

## Vector Storage Layer

A PostgreSQL instance extended with pgvector serves as the vector index. The system defines:

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE log_embeddings (
    id SERIAL PRIMARY KEY,
    log_text TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW()
);

This enables efficient similarity search via:

embedding <=> query_vector

which performs cosine or Euclidean distance retrieval depending on configuration.

This component is essential for enabling RAG over historical debugging logs, mapping user queries to semantically similar failure traces.

---

## Semantic Analytics Layer

### Clustering

The Streamlit application loads rag_interactions.parquet and computes:

- KMeans clusters (with dynamic K)
- Failure-weighted semantics
- Representative exemplar prompts per cluster

This produces a semantic taxonomy of system behaviors, useful for:

- Failure mode discovery
- UX pattern analysis
- Workload categorization

---

### Dimensionality Reduction

Visual cluster coherence is assessed using:

- Principal Component Analysis (PCA)
- UMAP manifold learning

These enable 2D embeddings suitable for operator dashboards.

UMAP is particularly effective due to its preservation of local neighborhood structure, supporting analysis of subtle failure sub-modes.

---

## LLM Debugging Agent Layer

The debugging interface uses a hybrid RAG + agentic specialization architecture.

### Retrieval

User queries are embedded via Gemini and matched against pgvector logs. Retrieved logs form contextual evidence for diagnosis.

---

### Agent Specialization

The orchestrator() directs queries to specialized agents using lightweight query classification:

Agent | Trigger domain
----- | --------------
failure_agent | crashes, exceptions
performance_agent | latency, timeouts
general_agent | fallback

This approach approximates a multi-agent tool routing scheme without requiring heavy frameworks.

---

### Response Format

The agent enforces a structured diagnostic schema:

Summary:
Causes:
Fix:
Verification:

This deviates from conversational LLM responses in favor of reproducible troubleshooting guidance.

---

## Dashboard and Observability Interface

The Streamlit UI exposes all upstream artifacts:

Visualization | Purpose
------------- | -------
Failures Over Time | Longitudinal reliability
Latency per Model | Performance analysis
Top Clusters | Semantic workload taxonomy
PCA / UMAP | Cluster structure exploration
RAG Debugging | On-demand semantic diagnosis

This satisfies core LLMOps observability goals including:

- descriptive analytics
- semantic telemetry
- clustering-based failure introspection
- direct debugging workflow support
