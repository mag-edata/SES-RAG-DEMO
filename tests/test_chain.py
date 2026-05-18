"""
chain.py のユニットテスト。
テストケース:C-01, C-02, C-03
"""
from unittest.mock import MagicMock, patch

from rag.chain import build_prompt, generate_reasons

# 固定テストデータ — 値は data/*.json の eng_001 / eng_002 および job_001 から派生
SAMPLE_CANDIDATES = [
    {
        "id": "eng_001",
        "name": "eng_001",
        "skills": "Java, Spring Boot, Oracle DB, Oracle SQL, REST API, Git, Maven, JUnit, マイクロサービス, Linux",
        "experience_years": 8.5,
        "score": 0.85,
        "summary": "Javaをメインとしたバックエンド開発者。金融・保険系の大規模システム開発経験が豊富。Spring Bootによるマイクロサービス設計・実装を得意とし、既存レガシーシステムのモダナイズ案件も多数経験。",
    },
    {
        "id": "eng_002",
        "name": "eng_002",
        "skills": "AWS, Terraform, AWS CDK, Linux, Python, Docker, CloudWatch, CI/CD, シェルスクリプト",
        "experience_years": 5.0,
        "score": 0.72,
        "summary": "AWSインフラの設計・構築・運用を専門とするクラウドエンジニア。IaCツール（Terraform/CDK）を活用したインフラ自動化と、コスト最適化を得意とする。監視・アラート設計の経験も豊富。",
    },
]

JOB_TEXT = "大規模な勘定系システムのリプレイスプロジェクト。Spring BootによるREST API開発およびマイクロサービス化を担当。既存COBOLシステムの解析・移行も含む。"


# C-01:build_prompt が空でない文字列を返却する
def test_build_prompt_returns_nonempty_string():
    prompt = build_prompt(JOB_TEXT, SAMPLE_CANDIDATES)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# C-02:build_prompt の出力に案件要件テキストと候補エンジニア名が含まれる
def test_build_prompt_contains_job_text_and_candidate_names():
    prompt = build_prompt(JOB_TEXT, SAMPLE_CANDIDATES)
    assert "勘定系システム" in prompt
    assert "eng_001" in prompt
    assert "eng_002" in prompt


# C-03:LLMをモック化した場合に generate_reasons が文字列を返却する
def test_generate_reasons_returns_string_with_mocked_llm():
    mock_response = MagicMock()
    mock_response.content = (
        "Candidate 1 (Test Engineer A) has hands-on Python and AWS experience, making them a strong fit.\n"
        "Candidate 2 (Test Engineer B) is a Java veteran; note the language difference."
    )

    with patch("rag.chain.ChatOpenAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        result = generate_reasons(JOB_TEXT, SAMPLE_CANDIDATES)

    assert isinstance(result, str)
    assert len(result) > 0
