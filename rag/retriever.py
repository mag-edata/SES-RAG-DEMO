"""Similarity search over the ChromaDB engineer index."""

import os

import chromadb
from openai import OpenAI

from rag.embedder import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_collection(collection_name: str = COLLECTION_NAME):
    """Open and return an existing ChromaDB collection.

    Args:
        collection_name: Name of the collection to retrieve.

    Returns:
        A ChromaDB Collection object.
    """
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return chroma.get_collection(collection_name)


def search_engineers(query_text: str, top_k: int = 3) -> list[dict]:
    """Embed a query string and retrieve the closest engineer documents from ChromaDB.

    Args:
        query_text: Free-text query describing job requirements or desired skills.
        top_k: Number of nearest-neighbor results to return.

    Returns:
        A list of dicts, each containing id, document, distance, and metadata
        for a matched engineer.
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


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return the top-k matching engineers for a job query, with similarity scores.

    Wraps search_engineers and normalises the cosine distance into a similarity
    score in the range [0, 1] (higher is more similar).

    Args:
        query: Job description or requirement text used as the search query.
        k: Maximum number of candidates to return.

    Returns:
        A list of dicts with keys id, name, summary, skills, experience_years,
        and score, sorted by descending similarity.
    """
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
