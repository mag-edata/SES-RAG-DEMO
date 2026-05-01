import json
import os

import chromadb
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "engineers"
ENGINEERS_DATA_PATH = "./data/engineers.json"


def load_engineers(filepath: str = ENGINEERS_DATA_PATH) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_profile_text(engineer: dict) -> str:
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
