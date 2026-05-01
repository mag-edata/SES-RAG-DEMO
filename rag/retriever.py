import os

import chromadb
from openai import OpenAI

from rag.embedder import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_collection(collection_name: str = COLLECTION_NAME):
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return chroma.get_collection(collection_name)


def search_engineers(query_text: str, top_k: int = 3) -> list[dict]:
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


def retrieve(query: str, k: int = 3) -> list[dict]:
    raw = search_engineers(query, top_k=k)
    return [
        {
            "id": r["id"],
            "name": r["metadata"].get("name", ""),
            "summary": r["document"],
            "skills": r["metadata"].get("skills", ""),
            "experience_years": r["metadata"].get("experience_years", 0),
            "score": max(0.0, 1 - r["distance"]),
        }
        for r in raw
    ]
