import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="SES Engineer Matching (RAG Demo)", layout="wide")
st.title("SES Engineer Matching (RAG Demo)")
st.caption("案件要件を入力すると、RAGを用いて最適なエンジニア候補をTop-3で提示します。")

# ──────────────────────────────
# Sidebar: Index build
# ──────────────────────────────
with st.sidebar:
    st.header("Admin Menu")
    st.write("初回利用前にインデックスを構築してください。")

    if st.button("Build Index"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("API key is not configured. Please check your .env file.")
        else:
            try:
                from rag.embedder import build_index
                with st.spinner("インデックスを構築中..."):
                    count = build_index()
                st.success(f"インデックス構築完了：{count} 件登録")
            except FileNotFoundError:
                st.error("Failed to load data file. Please check the file path.")
            except Exception as e:
                st.error(f"API call failed. Please wait a moment and try again.\n{e}")

    # Show current record count
    try:
        import chromadb
        chroma = chromadb.PersistentClient(path="./chroma_db")
        col = chroma.get_collection("engineers")
        st.metric("登録済みエンジニア数", col.count())
    except Exception:
        st.metric("登録済みエンジニア数", 0)

# ──────────────────────────────
# Main: Run matching
# ──────────────────────────────
job_text = st.text_area(
    "案件要件を入力してください",
    height=150,
    placeholder=(
        "例: Pythonを使ったWebバックエンド開発。"
        "FastAPIとAWSの経験必須。チームリード経験があれば尚可。"
        "3ヶ月以上の参画を想定。"
    ),
)

if st.button("マッチング実行", type="primary"):
    if not job_text.strip():
        st.error("Please enter job requirements.")
    elif not os.environ.get("OPENAI_API_KEY"):
        st.error("API key is not configured. Please check your .env file.")
    else:
        try:
            from rag.chain import run_rag
            with st.spinner("マッチング中..."):
                results = run_rag(job_text)

            if not results:
                st.error("No matching engineers found. Please try different requirements.")
            else:
                st.subheader("マッチング結果 Top 3")
                for i, result in enumerate(results, 1):
                    eng = result["engineer"]
                    with st.expander(
                        f"候補 {i}：{eng['name']}　（類似度スコア: {eng['score']:.3f}）",
                        expanded=True,
                    ):
                        st.write(f"**スキル:** {eng['skills']}")
                        st.write(f"**経験年数:** {eng['experience_years']} 年")
                        st.markdown(f"**マッチング理由:**\n\n{result['reason']}")

        except Exception as e:
            err = str(e)
            if "does not exist" in err or "No collection" in err:
                st.error("Index not built. Please build it from the sidebar.")
            elif "AuthenticationError" in err or "api_key" in err.lower():
                st.error("API key is not configured. Please check your .env file.")
            else:
                st.error(f"API call failed. Please wait a moment and try again.\n{err}")
