"""
embedder.py(インデックス構築)と retriever.py(類似度検索)の結合テスト。
テストケース:E-03, E-04, R-01, R-02, R-03, R-04
"""
import chromadb
import pytest
from unittest.mock import MagicMock, patch

from rag.embedder import build_index
from rag.retriever import search_engineers

# top_k=3 を満たすため data/engineers.json の3件(eng_001〜eng_003)を使用
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

# モック化したOpenAI Embedding APIが返却する固定1536次元ベクトル
FAKE_EMBEDDING = [0.1] * 1536


@pytest.fixture
def chroma_dir(tmp_path, monkeypatch):
    """テスト隔離のため、ChromaDBの保存先を一時ディレクトリにリダイレクトする。"""
    path = str(tmp_path / "chroma_test")
    monkeypatch.setattr("rag.embedder.CHROMA_PERSIST_DIR", path)
    monkeypatch.setattr("rag.retriever.CHROMA_PERSIST_DIR", path)
    return path


@pytest.fixture
def mock_openai():
    """embedder と retriever のOpenAIクライアントをパッチし、FAKE_EMBEDDING を返却させる。"""
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
    """retriever テスト用に SAMPLE_ENGINEERS からChromaDBインデックスを事前構築する。"""
    build_index(SAMPLE_ENGINEERS)
    return chroma_dir


# ──────────────────────────────────────────────────────────────────────────────
# E-03:build_index() が登録件数を返却する
# ──────────────────────────────────────────────────────────────────────────────
def test_build_index_returns_record_count(chroma_dir, mock_openai):
    count = build_index(SAMPLE_ENGINEERS)
    assert count == len(SAMPLE_ENGINEERS)


# ──────────────────────────────────────────────────────────────────────────────
# E-04:build_index() を2回実行しても重複登録されない
# ──────────────────────────────────────────────────────────────────────────────
def test_build_index_twice_no_duplicates(chroma_dir, mock_openai):
    build_index(SAMPLE_ENGINEERS)
    count = build_index(SAMPLE_ENGINEERS)
    assert count == len(SAMPLE_ENGINEERS)

    chroma = chromadb.PersistentClient(path=chroma_dir)
    col = chroma.get_collection("engineers")
    assert col.count() == len(SAMPLE_ENGINEERS)


# ──────────────────────────────────────────────────────────────────────────────
# R-01:search_engineers() がリストを返却する
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_returns_list(built_index):
    results = search_engineers("Python backend developer")
    assert isinstance(results, list)


# ──────────────────────────────────────────────────────────────────────────────
# R-02:top_k=3 で3件返却される
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_returns_top_k_records(built_index):
    results = search_engineers("Python backend developer", top_k=3)
    assert len(results) == 3


# ──────────────────────────────────────────────────────────────────────────────
# R-03:結果の各要素に id / document / distance / metadata が含まれる
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_result_has_required_keys(built_index):
    results = search_engineers("Python backend developer", top_k=1)
    assert all(key in results[0] for key in ("id", "document", "distance", "metadata"))


# ──────────────────────────────────────────────────────────────────────────────
# R-04:すべての distance が ChromaDB のコサイン距離範囲 [0.0, 2.0] に収まる
# ──────────────────────────────────────────────────────────────────────────────
def test_search_engineers_distance_in_valid_range(built_index):
    results = search_engineers("Python backend developer", top_k=3)
    for r in results:
        # ChromaDB のコサイン距離は理論上 [0, 2] だが、同一ベクトル間では
        # 浮動小数点演算により 0 をわずかに下回る値(〜-1e-5)が生じうる。
        assert -1e-4 <= r["distance"] <= 2.0
