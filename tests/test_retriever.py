"""
Integration tests for embedder.py (index build) and retriever.py (similarity search).
Test cases: E-03, E-04, R-01, R-02, R-03, R-04
"""
import chromadb
import pytest
from unittest.mock import MagicMock, patch

from rag.embedder import build_index
from rag.retriever import search_engineers

# 3 records from data/engineers.json (eng_001–eng_003) used to satisfy top_k=3
SAMPLE_ENGINEERS = [
    {
        "id": "eng_001",
        "氏名": "eng_001",
        "年齢": 32,
        "経験年数": 8.5,
        "所属": "プロパー",
        "参画可能時期": "即日",
        "概要": "Javaをメインとしたバックエンド開発者。金融・保険系の大規模システム開発経験が豊富。Spring Bootによるマイクロサービス設計・実装を得意とし、既存レガシーシステムのモダナイズ案件も多数経験。",
        "スキル": ["Java", "Spring Boot", "Oracle DB", "Oracle SQL", "REST API", "Git", "Maven", "JUnit", "マイクロサービス", "Linux"],
        "資格": ["Oracle Java SE 11 Silver", "AWS Solutions Architect Associate"],
        "勤務形態": "常駐・リモート両可",
        "希望勤務地": "東京・神奈川",
        "経歴": [
            "大手保険会社の保険金計算システムリプレイス（Java/Spring Boot、3年）",
            "証券会社の注文管理システムAPI開発（Java/REST API、2年）",
            "銀行の顧客管理システム設計・実装（Java/Oracle、1年）",
        ],
        "現在単価": "75万円/月",
    },
    {
        "id": "eng_002",
        "氏名": "eng_002",
        "年齢": 28,
        "経験年数": 5.0,
        "所属": "BP",
        "参画可能時期": "20260415",
        "概要": "AWSインフラの設計・構築・運用を専門とするクラウドエンジニア。IaCツール（Terraform/CDK）を活用したインフラ自動化と、コスト最適化を得意とする。監視・アラート設計の経験も豊富。",
        "スキル": ["AWS", "Terraform", "AWS CDK", "Linux", "Python", "Docker", "CloudWatch", "CI/CD", "シェルスクリプト"],
        "資格": ["AWS Solutions Architect Professional", "AWS DevOps Engineer Professional"],
        "勤務形態": "リモート希望",
        "希望勤務地": "フルリモート",
        "経歴": [
            "ECプラットフォームのAWSマルチアカウント設計・構築（Terraform、2年）",
            "SaaSサービスのインフラコスト30%削減（AWS Cost Optimization、1年）",
            "オンプレからAWSへの移行プロジェクト（2年）",
        ],
        "現在単価": "65万円/月",
    },
    {
        "id": "eng_003",
        "氏名": "eng_003",
        "年齢": 30,
        "経験年数": 6.5,
        "所属": "プロパー",
        "参画可能時期": "20260701",
        "概要": "Pythonによるデータ基盤・ETLパイプライン構築を専門とするデータエンジニア。GCP/BigQueryを活用したデータウェアハウス構築の経験が豊富。製造業・小売業でのデータ活用支援実績あり。",
        "スキル": ["Python", "BigQuery", "Airflow", "dbt", "BigQuery SQL", "GCP", "Pandas", "ETL", "Looker", "Linux"],
        "資格": ["Google Cloud Professional Data Engineer"],
        "勤務形態": "リモート可",
        "希望勤務地": "東京・フルリモート",
        "経歴": [
            "製造業の工場センサーデータ基盤構築（Python/BigQuery/Airflow、2年）",
            "小売チェーンのデータウェアハウス設計（dbt/BigQuery、2年）",
            "流通企業のリアルタイムデータパイプライン構築（Pub/Sub/Dataflow、1年）",
        ],
        "現在単価": "68万円/月",
    },
]

# Fixed 1536-dim vector returned by the mocked OpenAI Embedding API
FAKE_EMBEDDING = [0.1] * 1536


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    """Redirect ChromaDB storage to a temp directory for test isolation."""
    path = str(tmp_path / "chroma_test")
    monkeypatch.setattr("rag.embedder.CHROMA_PERSIST_DIR", path)
    monkeypatch.setattr("rag.retriever.CHROMA_PERSIST_DIR", path)
    return path


@pytest.fixture
def mock_openai():
    """Patch the OpenAI client in both embedder and retriever to return FAKE_EMBEDDING."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=FAKE_EMBEDDING)]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    with (
        patch("rag.embedder.OpenAI", return_value=mock_client),
        patch("rag.retriever.OpenAI", return_value=mock_client),
    ):
        yield mock_client


@pytest.fixture
def built_index(chroma_dir, mock_openai):
    """Pre-build a ChromaDB index from SAMPLE_ENGINEERS for retriever tests."""
    build_index(SAMPLE_ENGINEERS)
    return chroma_dir


# ──────────────────────────────────────────────────────────────────────────────
# E-03: build_index() returns the registered record count
# ──────────────────────────────────────────────────────────────────────────────
def test_build_index_returns_record_count(chroma_dir, mock_openai):
    count = build_index(SAMPLE_ENGINEERS)
    assert count == len(SAMPLE_ENGINEERS)


# ──────────────────────────────────────────────────────────────────────────────
# E-04: Running build_index() twice does not create duplicate entries
# ──────────────────────────────────────────────────────────────────────────────
def test_build_index_twice_no_duplicates(chroma_dir, mock_openai):
    build_index(SAMPLE_ENGINEERS)
    count = build_index(SAMPLE_ENGINEERS)
    assert count == len(SAMPLE_ENGINEERS)

    chroma = chromadb.PersistentClient(path=chroma_dir)
    col = chroma.get_collection("engineers")
    assert col.count() == len(SAMPLE_ENGINEERS)


# ──────────────────────────────────────────────────────────────────────────────
# R-01: search_engineers() returns a list
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_returns_list(built_index):
    results = search_engineers("Python backend developer")
    assert isinstance(results, list)


# ──────────────────────────────────────────────────────────────────────────────
# R-02: top_k=3 returns exactly 3 records
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_returns_top_k_records(built_index):
    results = search_engineers("Python backend developer", top_k=3)
    assert len(results) == 3


# ──────────────────────────────────────────────────────────────────────────────
# R-03: each result element contains id / document / distance / metadata
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_result_has_required_keys(built_index):
    results = search_engineers("Python backend developer", top_k=1)
    assert all(key in results[0] for key in ("id", "document", "distance", "metadata"))


# ──────────────────────────────────────────────────────────────────────────────
# R-04: all distance values fall within ChromaDB cosine distance range [0.0, 2.0]
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_distance_in_valid_range(built_index):
    results = search_engineers("Python backend developer", top_k=3)
    for r in results:
        # ChromaDB cosine distance is theoretically [0, 2], but floating-point
        # arithmetic can produce values slightly below 0 (~-1e-5) for identical vectors.
        assert -1e-4 <= r["distance"] <= 2.0
