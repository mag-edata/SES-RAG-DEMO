"""ChromaDBエンジニアインデックスに対する類似度検索。"""

import os

import chromadb
from openai import OpenAI

from rag.embedder import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

TOP_K = 3


def get_collection(collection_name: str = COLLECTION_NAME):
    """既存のChromaDBコレクションを開いて返却する。

    Args:
        collection_name: 取得するコレクションの名前。

    Returns:
        ChromaDB の Collection オブジェクト。
    """
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return chroma.get_collection(collection_name)


def search_engineers(query_text: str, top_k: int = TOP_K) -> list[dict]:
    """クエリ文字列をEmbedding化し、ChromaDBから最も近いエンジニアドキュメントを取得する。

    Args:
        query_text: 案件要件や希望スキルを記述した自由形式のクエリ。
        top_k: 返却する最近傍結果の件数。

    Returns:
        マッチしたエンジニアごとに id・document・distance・metadata を含む
        辞書のリスト。
    """
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=query_text)
    query_embedding = response.data[0].embedding

    collection = get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return [
        {
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """案件クエリに対する上位k件のマッチエンジニアを類似度スコア付きで返却する。

    search_engineers をラップし、コサイン距離を [0, 1] の範囲の類似度スコアへ
    正規化する(値が大きいほど類似度が高い)。

    Args:
        query: 検索クエリとして使用する案件説明または要件テキスト。
        k: 返却する候補の最大件数。

    Returns:
        id・name・summary・skills・experience_years・score をキーに持つ
        辞書のリスト。類似度の降順でソート済み。
    """
    raw = search_engineers(query, top_k=k)
    return [
        {
            "id": r["id"],
            "name": r["metadata"].get("name", ""),
            "summary": r["document"],
            "skills": r["metadata"].get("skills", ""),
            "experience_years": r["metadata"].get("experience_years", 0),
            "score": min(1.0, max(0.0, 1 - r["distance"])),
        }
        for r in raw
    ]
