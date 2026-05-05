"""Streamlit entry point: job requirement input, index build, and RAG-based engineer matching UI."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="IT Engineer & Job Requirement Matching (RAG Demo)", layout="wide")
st.title("IT Engineer & Job Requirement Matching (RAG Demo)")
st.caption("案件要件を入力すると、RAGを用いて最適なエンジニア候補Top-3を提示します。")

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
        "例："
        "タイトル: バックエンドエンジニア（Java/Spring Boot）"
        "クライアント: 大手金融システム会社A"
        "案件概要: 大規模な勘定系システムのリプレイスプロジェクト。Spring BootによるREST API開発およびマイクロサービス化を担当。既存COBOLシステムの解析・移行も含む。"
        "Mustスキル: [Java, Spring Boot, REST API, Oracle SQL, Git]"
        "Wantスキル: [COBOL, Oracle DB, マイクロサービス, AWS]"
        "最低経験年数: 3"
        "勤務形態: 常駐+リモート併用"
        "勤務地: 東京都千代田区"
        "期間: 20260401〜長期"
        "工程: [基本設計, 詳細設計, 製造, 単体テスト, 結合テスト]"
        "再委託: 不可"
        "契約: 準委任"
        "募集人数: 2"
        "想定単価幅: 60〜78万円/月"
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
