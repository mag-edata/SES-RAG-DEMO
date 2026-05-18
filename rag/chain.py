"""RAGチェイン:候補エンジニアを検索し、LLMでマッチング理由を生成する。"""

import os
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from rag.retriever import TOP_K, retrieve

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

For each candidate engineer, explain in 2-3 bullet points (in Japanese) how they align with the job requirements.
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
    """案件要件と候補リストをLLM向けのプロンプト文字列に整形する。

    Args:
        job_text: 案件説明または要件のテキスト原文。
        candidates: retrieve() が返却する候補辞書のリスト。

    Returns:
        LLMにそのまま送信可能な整形済みプロンプト文字列。
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
    """LLM出力を [Candidate N] マーカーで分割し、候補ごとの理由文字列に変換する。

    Args:
        text: 構造化された [Candidate N] セクションを含むLLM応答テキスト原文。
        n: 抽出を期待する候補数。

    Returns:
        長さnのリスト。各要素は対応する候補の理由テキスト。
        該当セクションが見つからない場合は空文字列を返す。
    """
    reasons = []
    for i in range(1, n + 1):
        pattern = rf'\[Candidate {i}\](.*?)(?=\[Candidate {i + 1}\]|$)'
        match = re.search(pattern, text, re.DOTALL)
        reasons.append(match.group(1).strip() if match else "")
    return reasons


def generate_reasons(job_text: str, candidates: list[dict]) -> str:
    """LLMを呼び出して、各候補エンジニアのマッチング理由を生成する。

    Args:
        job_text: LLMのコンテキストとして使用する案件説明。
        candidates: retrieve() が返却する候補辞書のリスト。

    Returns:
        候補ごとの説明を含むLLM応答テキスト原文。
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


def run_rag(job_description: str, k: int = TOP_K) -> list[dict]:
    """RAGパイプライン全体を実行し、候補取得と理由生成をまとめて行う。

    アプリケーションのメインエントリーポイント。意味検索でマッチング上位k件の
    エンジニアを取得し、各結果に対してLLMが生成したマッチング理由を付与する。

    Args:
        job_description: エンジニアとマッチングする案件要件の自由形式テキスト。
        k: 取得および説明を生成する候補エンジニアの件数。

    Returns:
        以下のキーを持つ辞書のリスト:
            - engineer: retrieve() が返却する候補辞書
            - reason: 当該候補に対するLLM生成の説明文字列
    """
    candidates = retrieve(job_description, k=k)
    reason_text = generate_reasons(job_description, candidates)
    per_candidate_reasons = _parse_per_candidate_reasons(reason_text, len(candidates))
    return [
        {"engineer": c, "reason": reason}
        for c, reason in zip(candidates, per_candidate_reasons)
    ]
