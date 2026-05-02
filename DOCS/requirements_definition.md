# Requirements Definition

**Project Name:** SES-RAG-DEMO  
**Version:** 1.0  
**Date Created:** 2026-03-20

---

## 1. Purpose and Background

- In the SES (System Engineering Services) industry, matching job requirements with engineers is often performed manually by sales, leading to over-reliance on individuals and operational inefficiency.
- This system leverages RAG (Retrieval-Augmented Generation) technology to achieve matching based on semantic similarity between job requirements and engineer skill profiles.
- It serves as a demo showcasing "AI-assisted decision support" for hiring managers and recruitment agents.
- The primary goal is not to maximize matching accuracy, but to demonstrate the ability to design and reason about RAG systems.

## 2. Problem Definition (Current Challenges in SES Matching)

- **Current state:** Sales manually compare engineer skill sheets against job requirements using spreadsheets or visual inspection.
- **Problem 1 (Search precision):** Keyword-based exact matching cannot handle variations in expression (e.g., "Python engineer" vs. "Python development experience"), causing missed matching opportunities.
- **Problem 2 (Explainability):** Articulating the rationale for why a particular candidate was recommended incurs high cognitive cost.
- **Problem 3 (Scalability):** As the number of candidate engineers grows, the cost of manual comparison increases linearly.

This system addresses Problems 1 and 3 via semantic search using Embeddings, and Problem 2 via LLM-generated reasoning.

## 3. System Overview

- Given a job requirement text as input, the system retrieves the top-k semantically similar engineer candidates and automatically generates matching rationale for each using an LLM.
- Backend: LangChain + ChromaDB + OpenAI API
- Frontend: Streamlit (runs in a web browser)
- Data: 20 fictional job requirements (`jobs.json`) and 20 fictional engineers (`engineers.json`)
- Deployment: Streamlit Cloud (connected to GitHub)

## 4. Functional Requirements

### 4.1 Engineer Search Feature

- Vectorize the input job requirement text using the OpenAI Embedding API.
- Compute similarity against engineer profile vectors stored in ChromaDB.
- Return the top k engineers (default: 3) ranked by cosine similarity.
- Search results include engineer name, skills, and similarity score.

### 4.2 Matching Reason Generation Feature

- Incorporate the top-k candidate engineers' information and the job requirement into a prompt.
- Submit to gpt-4o-mini to generate matching rationale per candidate.
- Rationale is presented as 2-3 bullet points with explicit supporting evidence.
- Presented as explanatory text to assist human decision-making, not as an automated hiring decision.

### 4.3 Index Build Feature

- Load `engineers.json` and convert each engineer's profile to Embeddings.
- Register the converted vectors into a ChromaDB collection (supports both initial build and overwrite).
- Executable via a button in the Streamlit sidebar.
- Display the record count and result in the UI after completion.

## 5. Non-Functional Requirements

### Performance

| Process | Target |
|---------|--------|
| Search (Embedding generation + ChromaDB query) | Within 3 seconds |
| LLM-based reason generation (gpt-4o-mini) | Within 10 seconds |
| Index build (20 records) | Within 30 seconds |

### Security

- OpenAI API key must be managed via environment variables (`.env`) and must not be hardcoded.
- The `.env` file must be added to `.gitignore` to prevent GitHub commits.
- Use `st.secrets` for secret management on Streamlit Cloud.

### Availability

- Accept constraints of the Streamlit Cloud free plan (e.g., sleep mode).
- Ensure equivalent operation in local environments.

## 6. Constraints

| Item | Detail |
|------|--------|
| Language | Python 3.14 |
| LLM | OpenAI gpt-4o-mini (mini selected to minimize cost) |
| Embedding | text-embedding-3-small (same reason) |
| Vector DB | ChromaDB (local file storage) |
| Data volume | 20 job requirements, 20 engineers (fixed for demo) |
| Budget | Within the free tier of OpenAI API and Streamlit Cloud free plan |
| Sample data | Must not include real company names or real individual names (fictional only) |

## 7. Glossary

| Term | Definition |
|------|-----------|
| RAG (Retrieval-Augmented Generation) | A technique that retrieves external knowledge to augment LLM generation. |
| Embedding | The process of converting text into a numerical vector that preserves semantic meaning. |
| ChromaDB | An open-source vector database that runs locally. |
| Top-k | A search configuration that returns the top k results by similarity score. Default in this system is 3. |
| SES (System Engineering Services) | A form of IT staffing where engineers are dispatched to client sites under secondment or quasi-mandate contracts. |
| Cosine Similarity | A metric that measures semantic closeness between two vectors using the angle between them (range: 0 to 1). |
