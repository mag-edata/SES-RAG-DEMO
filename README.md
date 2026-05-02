# SES-RAG-DEMO

A RAG-based decision support system that semantically matches System Engineering Service (SES) job requirements with engineer profiles.

---

## Problem

In SES sales, matching job requirements with engineer profiles is a time-consuming and highly subjective process.

### core bottlenecks:

- **High search cost**
  Salespersons spend significant time identifying potential candidates from large pools of engineer profiles.

- **Skill-dependent decision quality**
  The quality of matching depends heavily on individual experience, leading to inconsistent outcomes.

- **Lack of explainability**
  Even when a candidate is selected, reasoning is often implicit and difficult to standardize or share.

As a result:
- Slow candidate shortlisting
- High variance in matching quality
- Limited reproducibility of successful decisions

---

## Solution

This system reframes matching as a **decision-support problem**, not just a search problem.

Using Retrieval-Augmented Generation (RAG), it:

1. **Reduces search time**  
   Retrieves top candidate engineers based on semantic similarity

2. **Standardizes decision-making**  
   Generates structured reasoning for why each candidate matches the requirements

3. **Improves reproducibility**  
   Makes implicit judgment criteria explicit and shareable

---

## Architecture

This system uses Retrieval-Augmented Generation (RAG) to:

1. Embed job requirements and engineer profiles into vector space
2. Retrieve top-k candidates by semantic similarity
3. Generate reasoning for each match using LLM

```
[Job Requirements]
        ↓ Embedding
[Vector DB (Chroma)] ← [Engineer Profiles]
        ↓ Top-k Retrieval
[LLM Reasoning Layer]
        ↓
[Top Matches + Explanation]
```

Example:

Input (Job Requirement)
- Backend engineer (Java)
- AWS experience
- Microservices architecture

Output (Top Match)

**Engineer A**

**Why matched:**
- 3+ years AWS (EC2, Lambda)
- Designed microservices-based systems
- Experience in scalability optimization

→ Not keyword overlap, but **contextual alignment in system design experience**

---

## Evaluation (Prototype)

Manual evaluation pending — qualitative observations only.

Automated accuracy metrics (e.g. Top-3 precision) are planned for a future phase.
Current validation: unit tests (12 passing) covering embedding, retrieval, and chain logic.

---

## Design Decisions

### Why RAG (not keyword search)

- Captures semantic similarity beyond exact terms
- Handles implicit skill representation
- Enables reasoning layer (not just retrieval)


### Why Top-3

- Matches real salesperson workflow (shortlist candidates)
- Trade-off between recall and decision cost


### Why text-embedding-3-small

- Sufficient semantic performance for prototype
- Lower cost → faster iteration

### Why explanation matters

In SES sales, selecting a candidate is not enough — the salesperson must also explain *why*.

This system externalizes that reasoning, enabling:
- Knowledge sharing across team members
- Reduced dependency on individual experience

---

## Limitations

- Small dataset (20 × 20)
- No automated evaluation metrics (manual only)
- Sensitive to prompt quality

---

## Tech Stack

- **LLM:** OpenAI API (gpt-4o-mini)
- **Embedding:** OpenAI text-embedding-3-small
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **UI:** Streamlit
- **Language:** Python 3.14

---

## Key Takeaway

This project demonstrates:

- Semantic matching of unstructured text
- LLM-based reasoning on top of retrieval
- Designing AI systems as **decision-support infrastructure**, not just automation tools

---

## Demo

> [Streamlit Cloud リンクをここに追加]

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/mag-edata/SES-RAG-DEMO.git
cd SES-RAG-DEMO

# 2. Install dependencies
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run
streamlit run app.py
```

---

## Project Structure

```
SES-RAG-DEMO/
├── app.py                  # Streamlit entry point
├── rag/
│   ├── embedder.py         # Embedding generation & ChromaDB indexing
│   ├── retriever.py        # Semantic similarity search
│   └── chain.py            # LangChain RAG chain
├── data/
│   ├── jobs.json           # Sample job requirements (20 entries)
│   └── engineers.json      # Sample engineer profiles (20 entries)
├── tests/
│   ├── test_embedder.py    # Unit tests: embedding & indexing (E-01, E-02, E-05)
│   ├── test_retriever.py   # Integration tests: similarity search (E-03, E-04, R-01~R-04)
│   └── test_chain.py       # Unit tests: RAG chain (C-01, C-02, C-03)
├── DOCS/
│   ├── requirements_definition.md
│   ├── external_design.md
│   ├── internal_design.md
│   ├── test_design.md
│   └── test_cases.md
├── requirements.txt
├── .env.example
└── README.md
```

---

## Background

This project is built on my experience as a former SES sales representative turned Python/NLP engineer. The matching problem is one I observed firsthand — and this demo is my attempt to solve it with modern LLM technology.

The design focus is not just "how to implement RAG" but **"what decision should RAG support"** — treating the matching system as a decision-support tool rather than an automation.

---

## Author

**mag** — NLP/AI Engineer
Former SES sales → Python Engineer
