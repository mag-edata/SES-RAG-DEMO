"""エンジニアプロフィールのEmbedding生成とChromaDBインデックス管理。"""

import json
import os

import chromadb
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "engineers"
ENGINEERS_DATA_PATH = "./data/engineers.json"


def load_engineers(filepath: str = ENGINEERS_DATA_PATH) -> list[dict]:
    """JSONファイルからエンジニアプロフィールを読み込む。

    Args:
        filepath: エンジニアデータを含むJSONファイルのパス。

    Returns:
        エンジニアプロフィール辞書のリスト。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_profile_text(engineer: dict) -> str:
    """エンジニアプロフィール辞書をEmbedding入力用の単一テキストに直列化する。

    Args:
        engineer: 氏名・概要・スキル・経験年数・経歴などのキーを持つ辞書。

    Returns:
        エンジニアプロフィールを連結した文字列表現。
    """
    name = engineer.get("氏名", "")
    summary = engineer.get("概要", "")
    skills = ", ".join(engineer.get("スキル", []))
    experience = engineer.get("経験年数", "")
    history = " / ".join(engineer.get("経歴", []))
    return (
        f"名前: {name}  概要: {summary}  "
        f"スキル: {skills}  経験年数: {experience}年  経歴: {history}"
    )


def build_index(engineers: list[dict] = None, collection_name: str = COLLECTION_NAME) -> int:
    """全エンジニアプロフィールをEmbedding化しChromaDBコレクションに登録する。

    同名の既存コレクションを削除してから再構築するため、冪等に実行できる。

    Args:
        engineers: エンジニアプロフィール辞書のリスト。Noneの場合は
            デフォルトのJSONファイルから読み込む。
        collection_name: 作成するChromaDBコレクションの名前。

    Returns:
        インデックスに登録したエンジニアドキュメントの件数。
    """
    if engineers is None:
        engineers = load_engineers()

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 冪等性を保証するため、既存コレクションを削除して再構築する
    try:
        chroma.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    ids, documents, embeddings, metadatas = [], [], [], []
    for engineer in engineers:
        doc_text = build_profile_text(engineer)
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=doc_text)
        ids.append(engineer["id"])
        documents.append(doc_text)
        embeddings.append(response.data[0].embedding)
        metadatas.append({
            "name": engineer.get("氏名", ""),
            "skills": ", ".join(engineer.get("スキル", [])),
            "experience_years": float(engineer.get("経験年数", 0)),
            "availability": engineer.get("参画可能時期", ""),
        })

    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(ids)
