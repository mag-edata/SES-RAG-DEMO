# Internal Design Document

**Project Name:** SES-RAG-DEMO  
**Version:** 1.0  
**Date Created:** 2026-03-20

---

## 1. System Architecture Diagram

```
+----------------------------------------------------------+
|                  Streamlit (app.py)                      |
|  +----------------------+  +---------------------------+ |
|  |  Main Screen         |  |  Sidebar                  | |
|  |  (Job Input/Results) |  |  (Index Build)            | |
+--+----------------------+--+---------------------------+-+
   |                     |  |                          |
   | Search Query        |  | Build Trigger            |
   v                     |  v                          |
+------------------+     | +----------------------+    |
|  retriever.py    |     | |   embedder.py         |    |
|  Similarity      |     | |   Embedding / DB      |    |
|  Search          |     | |   Registration        |    |
+--------+---------+     | +----------+-----------+    |
         |               |            |               |
         | Top-k Results |            | Vector Insert  |
         v               |            v               |
+------------------+     | +----------------------+   |
|  chain.py        |     | |   ChromaDB            |   |
|  LLM-based       |     +-|   (Local File)        +---+
|  Reason Gen      |       +----------------------+
+--------+---------+                  ^
         | Reason Text                |
         v                            |
+------------------+    +---------------------------+
|  Streamlit UI    |    |   OpenAI API               |
|  Result Display  |    |   - Embeddings             |
+------------------+    |   - Chat Completions       |
                        +---------------------------+
```

**Data Flow Summary**

| Flow | Path |
|------|------|
| Index Build | `engineers.json` → `embedder.py` → OpenAI Embedding API → ChromaDB |
| Run Matching | Input text → OpenAI Embedding API → ChromaDB search → `chain.py` → LLM → UI display |

---

## 2. Module Structure

### 2.1 `rag/embedder.py`

**Role:** Generate Embeddings from engineer data and register them in ChromaDB

**Key Functions**

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_engineers` | `(filepath: str) -> list[dict]` | Load `engineers.json` and return as a list. |
| `build_profile_text` | `(engineer: dict) -> str` | Concatenate an engineer's dict fields into a single string (used as the search target text). Example: `"Name: Ichiro Tanaka  Skills: Python, AWS  Years of Experience: 5 ..."` |
| `build_index` | `(engineers: list[dict] = None, collection_name: str = "engineers") -> int` | Convert all engineer profile texts to Embeddings and register in ChromaDB. When `engineers` is omitted, calls `load_engineers()` internally. Deletes and rebuilds the existing collection (ensures idempotency). Returns number of records registered. |

**External Dependencies:** `openai.OpenAI`, `chromadb`, `json`, `os`

### 2.2 `rag/retriever.py`

**Role:** Similarity search for engineers based on job requirement text

**Key Functions**

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_collection` | `(collection_name: str = "engineers") -> chromadb.Collection` | Retrieve and return the ChromaDB collection. |
| `search_engineers` | `(query_text: str, top_k: int = 3) -> list[dict]` | Convert query text to an Embedding and search the collection for the top_k results. Returns `[{"id": str, "document": str, "distance": float, "metadata": dict}]`. |
| `retrieve` | `(query: str, k: int = 3) -> list[dict]` | Wrapper around `search_engineers()`. Converts ChromaDB cosine distance to a similarity score (`1 - distance`) and returns `[{"id", "name", "skills", "experience_years", "score", ...}]` ready for use by `chain.py`. |

**External Dependencies:** `openai.OpenAI`, `chromadb`, `rag.embedder`

### 2.3 `rag/chain.py`

**Role:** Matching reason generation via LangChain RAG Chain

**Key Functions**

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_prompt` | `(job_text: str, candidates: list[dict]) -> str` | Generate a prompt string for the LLM from job requirements and candidate engineer list. |
| `generate_reasons` | `(job_text: str, candidates: list[dict]) -> str` | Submit the prompt to gpt-4o-mini and return matching reason text. Uses LangChain's `ChatOpenAI` + `PromptTemplate`. |
| `run_rag` | `(job_description: str, k: int = 3) -> list[dict]` | **Main entry point.** Executes the full RAG pipeline: calls `retrieve()` then `generate_reasons()`, parses per-candidate reason text, and returns `[{"engineer": dict, "reason": str}]`. Called by `app.py`. |

**External Dependencies:** `langchain_openai.ChatOpenAI`, `langchain_core.prompts.PromptTemplate`, `os`

### 2.4 `app.py`

**Role:** Streamlit UI entry point

**Processing Flow**

1. Configure page settings and display title
2. Load environment variables (python-dotenv)
3. Sidebar: Index build button → call `embedder.build_index()`
4. Main: Text area input → Run Matching button
5. On button press: `chain.run_rag()` → `retriever.retrieve()` → `chain.generate_reasons()` → display results

**External Dependencies:** `streamlit`, `dotenv` (python-dotenv), `rag.embedder`, `rag.retriever`, `rag.chain`

---

## 3. Data Flow

### Index Build Flow

1. `app.py` (Sidebar): Receives button press event
2. `embedder.load_engineers()`: Loads `data/engineers.json` (20 records)
3. `embedder.build_profile_text()`: Converts each engineer to a text string
4. OpenAI Embeddings API: Text → 1536-dimensional vector (20 API calls)
5. ChromaDB: Registers vector + metadata + document into "engineers" collection
6. `app.py`: Displays record count in UI

### Matching Execution Flow

1. `app.py` (Main): Receives input text and button press
2. OpenAI Embeddings API: Input text → query vector (1 API call)
3. ChromaDB: Cosine similarity search → returns Top-3 results
4. `chain.build_prompt()`: Embeds job requirements + Top-3 candidates into prompt
5. OpenAI Chat Completions API: Prompt → reason text generation
6. `app.py`: Displays candidate engineer info + reason text in UI

---

## 4. ChromaDB Schema Design

**Collection Name:** `engineers`

**Field Structure**

| Field | Type | Description |
|-------|------|-------------|
| id | str | Engineer ID (from engineers.json) |
| document | str | Profile text (source text for Embedding) |
| embedding | vector | 1536-dimensional float (text-embedding-3-small) |
| metadata | dict | Supplementary fields (see below) |

**Metadata Fields**

| Key | Type | Description |
|-----|------|-------------|
| name | str | Engineer name |
| skills | str | Skill list (comma-separated string) |
| experience_years | int | Years of experience |
| availability | str | Availability status ("available" / "unavailable") |

**Persistence Configuration**

- Uses ChromaDB `PersistentClient` (stored in `./chroma_db/` directory)
- Automatically loads existing collection on app startup

---

## 5. Prompt Design

**System Prompt**

```
You are an expert in SES matching.
Based on the job requirements and candidate engineer profiles provided,
explain with supporting evidence why each engineer is a good fit for
this job requirement.
Your response is intended to assist human decision-making and does not
guarantee hiring outcomes.
```

**User Prompt Template**

```
[Job Requirements]
{job_text}

[Candidate Engineer List]
{candidates_text}

For each of the above candidate engineers, please explain in 2-3 bullet
points how they align with the job requirements.
```

**`candidates_text` Format**

```
Candidate 1: {name}
Skills: {skills}
Experience: {experience_years} years
Similarity score: {score}
```

---

## 6. Environment Variables and Configuration Values

**Environment Variables (`.env` file)**

| Variable Name | Required | Description |
|---------------|----------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |

**Configuration Constants (managed as constants in code)**

| Constant Name | Value | Description |
|---------------|-------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model used |
| `TOP_K` | `3` | Number of top search results |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature parameter |
| `LLM_MAX_TOKENS` | `800` | LLM max token count |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `COLLECTION_NAME` | `engineers` | Collection name |
| `ENGINEERS_DATA_PATH` | `./data/engineers.json` | Engineer data file path |
