# Guillermo Lupiáñez Tapia - Professional Profile

## Summary

Senior Backend Engineer with 5+ years of experience building high-throughput
systems with Python, Kafka and AWS. Experienced in distributed systems,
event-driven architectures, production microservices, multi-tenant platforms
and Applied AI capabilities. Based in Spain with EU work authorization. Open to
remote roles and EU relocation.

## Engineering focus

- Backend: Python, FastAPI, Flask, Django, REST, GraphQL
- Distributed systems: Apache Kafka, event-driven architecture, RabbitMQ
- Cloud and operations: AWS, Docker, Kubernetes, Helm, CI/CD, Grafana, Kibana
- Data: PostgreSQL, MongoDB, Redis
- Applied AI: MCP, RAG, Amazon Bedrock, production LLM integrations
- Quality: Pytest, test architecture, deployment reliability

## Experience

### Plénitas - Senior Backend Engineer

July 2025 - July 2026

Important distinction: Guillermo did not introduce Kafka as a replacement for
RabbitMQ. Kafka already existed in the original telemetry architecture. He
redesigned the path to publish directly to Kafka, removing RabbitMQ and its
connector from the critical path.

- Led the redesign of playback telemetry handling 100K+ sustained events/sec,
  publishing directly to Kafka and removing RabbitMQ from the critical path.
- Migrated managed Kafka from Confluent Cloud to Amazon MSK, reducing streaming
  costs by 80%, approximately from EUR 1,000 to EUR 200 per month.
- Built a Kafka-based viewing-progress service for a seven-figure user platform,
  using event coalescing to reduce Redis write pressure.
- Worked across a multi-tenant video platform serving approximately 50 tenants.
- Designed AWS media workflows spanning live ingestion, adaptive delivery,
  clipping, stream concatenation, transcoding and transcription.
- Built a multi-tenant MCP platform with token-derived isolation. The client ID
  from each token was propagated to tenant-scoped GraphQL queries and tools.
- Added human approval for write operations and retrieval over 100+ Confluence
  pages. Amazon Bedrock supported LLM-backed features in the wider product.

The managed Kafka platform was subsequently migrated from Confluent Cloud to
Amazon MSK.

### Deutsche Telekom - Python Backend Engineer

November 2023 - July 2025

- Developed and operated 10+ production microservices with Python, FastAPI and
  PostgreSQL on AWS, using Docker, Kubernetes and Helm.
- Owned releases from application versioning through environment promotion.
- Diagnosed and resolved deployment failures caused by incorrectly applied
  database migrations.
- Worked with GitLab CI/CD and production visibility through Grafana and Kibana.

### Workdeck - Full Stack Engineer

February 2023 - November 2023

- Built Python/FastAPI services and Angular features for a multi-tenant SaaS
  platform serving approximately 30 customer organizations.
- Delivered Shopify, LinkedIn and Google integrations, event-driven AWS
  workflows and containerized cloud deployments.

### nucleoo - Full Stack Engineer

January 2022 - February 2023

- Developed Flask/MongoDB APIs and Angular modules for client-facing and
  internal product workflows, including third-party integrations and Azure
  deployments.
- Redesigned Pytest and test-database architecture, increasing coverage from
  30% to 80%, removing flaky tests and stabilizing CI.

### CXPLUS - Junior Full Stack Developer

February 2021 - January 2022

- Helped build a cloud ERP/CRM for internet service providers from scratch with
  Django REST and PostgreSQL.
- Designed core data models and deployed the containerized product on GCP.

## Personal project

### MCP RAG Portfolio

This public portfolio demo is a personal project. It uses FastAPI, MCP,
ChromaDB, local sentence-transformer embeddings and a pluggable LLM layer with
Ollama, OpenAI and Amazon Bedrock providers. It is separate from the production
MCP implementation described in the Plénitas experience.

## Education

BSc in Computer Science, University of Granada, 2017-2021. Final thesis graded
9.8/10: a queueing-theory server simulation.

## Languages

- Spanish: native
- English: full professional proficiency, C1
- French: B1

## Target roles

Senior Backend Engineer roles where Python, distributed systems, Kafka and AWS
are central, with Applied AI where it solves a real product or operational
problem.
