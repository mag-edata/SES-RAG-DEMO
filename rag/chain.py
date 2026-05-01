"""RAG chain: retrieves candidate engineers and generates match reasoning via LLM."""

import os
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from rag.retriever import retrieve

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 800

_SYSTEM_PROMPT = (
    "You are an expert in SES matching. "
    "Based on the job requirements and candidate engineer profiles provided, "
    "explain with supporting evidence why each engineer is a good fit for this job. "
    "This explanation is intended to assist human decision-making and does not guarantee hiring outcomes."
)

_USER_TEMPLATE = """\
[Job Requirements]
{job_text}

[Candidate Engineer List]
{candidates_text}

For each candidate engineer, explain in 2-3 bullet points how they align with the job requirements.
Use exactly this format (do not omit any [Candidate N] header):

[Candidate 1]
- reason
- reason

[Candidate 2]
- reason
- reason

[Candidate 3]
- reason
- reason"""

_PROMPT = PromptTemplate(
    input_variables=["job_text", "candidates_text"],
    template=_USER_TEMPLATE,
)


def build_prompt(job_text: str, candidates: list[dict]) -> str:
    """Format job requirements and candidate list into the LLM prompt string.

    Args:
        job_text: Raw job description or requirement text.
        candidates: List of candidate dicts as returned by retrieve().

    Returns:
        A formatted prompt string ready to be sent to the LLM.
    """
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"Candidate {i}: {c['name']}")
        lines.append(f"Skills: {c['skills']}")
        lines.append(f"Experience: {c.get('experience_years', '')} years")
        lines.append(f"Similarity score: {c['score']:.3f}")
        lines.append("")
    candidates_text = "\n".join(lines)
    return _PROMPT.format(job_text=job_text, candidates_text=candidates_text)


def _parse_per_candidate_reasons(text: str, n: int) -> list[str]:
    """Split LLM output into per-candidate reason strings using [Candidate N] markers.

    Args:
        text: Raw LLM response text containing structured [Candidate N] sections.
        n: Expected number of candidates to extract.

    Returns:
        A list of length n where each element is the reason text for the
        corresponding candidate. Returns an empty string for any section not found.
    """
    reasons = []
    for i in range(1, n + 1):
        pattern = rf'\[Candidate {i}\](.*?)(?=\[Candidate {i + 1}\]|$)'
        match = re.search(pattern, text, re.DOTALL)
        reasons.append(match.group(1).strip() if match else "")
    return reasons


def generate_reasons(job_text: str, candidates: list[dict]) -> str:
    """Call the LLM to generate match reasoning for each candidate engineer.

    Args:
        job_text: Job description used as context for the LLM.
        candidates: List of candidate dicts as returned by retrieve().

    Returns:
        Raw LLM response text containing per-candidate explanations.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    prompt_text = build_prompt(job_text, candidates)
    response = llm.invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", prompt_text),
    ])
    return response.content


def run_rag(job_description: str, k: int = 3) -> list[dict]:
    """Execute the full RAG pipeline: retrieve candidates and generate reasoning.

    This is the main entry point for the application. It retrieves the top-k
    matching engineers via semantic search and augments each result with an
    LLM-generated explanation of why they fit the job.

    Args:
        job_description: Free-text job requirements to match against engineers.
        k: Number of candidate engineers to retrieve and explain.

    Returns:
        A list of dicts, each with keys:
            - engineer: candidate dict from retrieve()
            - reason: LLM-generated explanation string for this candidate
    """
    candidates = retrieve(job_description, k=k)
    reason_text = generate_reasons(job_description, candidates)
    per_candidate_reasons = _parse_per_candidate_reasons(reason_text, len(candidates))
    return [
        {"engineer": c, "reason": reason}
        for c, reason in zip(candidates, per_candidate_reasons)
    ]
