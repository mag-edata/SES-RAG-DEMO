"""Embedding generation and ChromaDB index management for engineer profiles."""

import json
import os

import chromadb
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "engineers"
ENGINEERS_DATA_PATH = "./data/engineers.json"


def load_engineers(filepath: str = ENGINEERS_DATA_PATH) -> list[dict]:
    """Load engineer profiles from a JSON file.

    Args:
        filepath: Path to the JSON file containing engineer data.

    Returns:
        A list of engineer profile dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_profile_text(engineer: dict) -> str:
    """Serialize an engineer profile dict into a single text string for embedding.

    Args:
        engineer: A dict with keys such as 氏名, 概要, スキル, 経験年数, and 経歴.

    Returns:
        A concatenated string representation of the engineer's profile.
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
    """Embed all engineer profiles and store them in a ChromaDB collection.

    Drops any existing collection with the same name before rebuilding, ensuring
    idempotent execution.

    Args:
        engineers: List of engineer profile dicts. Loads from the default JSON
            file when None.
        collection_name: Name of the ChromaDB collection to create.

    Returns:
        The number of engineer documents indexed.
    """
    if engineers is None:
        engineers = load_engineers()

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection and rebuild to ensure idempotency
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
