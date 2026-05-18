"""Streamlitエントリーポイント:案件要件入力・インデックス構築・RAGによるエンジニアマッチングUI。"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="ITエンジニア × 案件要件マッチング（RAGデモ）", layout="wide")
st.title("ITエンジニア × 案件要件マッチング（RAGデモ）")
st.caption("案件要件を入力すると、RAGを用いて最適なエンジニア候補Top-3を提示します。")

# ──────────────────────────────
# サイドバー:インデックス構築
# ──────────────────────────────
with st.sidebar:
    st.header("管理メニュー")
    st.write("初回利用前にインデックスを構築してください。")

    if st.button("インデックス構築"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("APIキーが設定されていません。.env ファイルを確認してください。")
        else:
            try:
                from rag.embedder import build_index
                with st.spinner("インデックスを構築中..."):
                    count = build_index()
                st.success(f"インデックス構築完了：{count} 件登録")
            except FileNotFoundError:
                st.error("データファイルの読み込みに失敗しました。ファイルパスを確認してください。")
            except Exception as e:
                st.error(f"API呼び出しに失敗しました。しばらく待ってから再試行してください。\n{e}")

    # 現在の登録件数を表示
    try:
        import chromadb
        chroma = chromadb.PersistentClient(path="./chroma_db")
        col = chroma.get_collection("engineers")
        st.metric("登録済みエンジニア数", col.count())
    except Exception:
        st.metric("登録済みエンジニア数", 0)

# ──────────────────────────────
# メイン:マッチング実行
# ──────────────────────────────
job_text = st.text_area(
    "案件要件を入力してください",
    height=400,
    placeholder=(
        "例：\n"
        "タイトル: バックエンドエンジニア（Java/Spring Boot）\n"
        "クライアント: 大手金融システム会社A\n"
        "案件概要: 大規模な勘定系システムのリプレイスプロジェクト。Spring BootによるREST API開発およびマイクロサービス化を担当。既存COBOLシステムの解析・移行も含む。\n"
        "Mustスキル: [Java, Spring Boot, REST API, Oracle SQL, Git]\n"
        "Wantスキル: [COBOL, Oracle DB, マイクロサービス, AWS]\n"
        "最低経験年数: 3\n"
        "勤務形態: 常駐+リモート併用\n"
        "勤務地: 東京都千代田区\n"
        "期間: 20260401〜長期\n"
        "工程: [基本設計, 詳細設計, 製造, 単体テスト, 結合テスト]\n"
        "再委託: 不可\n"
        "契約: 準委任\n"
        "募集人数: 2\n"
        "想定単価幅: 60〜78万円/月"
    ),
)

if st.button("マッチング実行", type="primary"):
    if not job_text.strip():
        st.error("案件要件を入力してください。")
    elif not os.environ.get("OPENAI_API_KEY"):
        st.error("APIキーが設定されていません。.env ファイルを確認してください。")
    else:
        try:
            from rag.chain import run_rag
            with st.spinner("マッチング中..."):
                results = run_rag(job_text)

            if not results:
                st.error("該当するエンジニアが見つかりませんでした。要件を変更して再度お試しください。")
            else:
                st.subheader("マッチング結果 Top 3")
                for i, result in enumerate(results, 1):
                    eng = result["engineer"]
                    with st.expander(
                        f"候補 {i}:{eng['name']}　(類似度スコア: {eng['score']:.3f})",
                        expanded=True,
                    ):
                        st.write(f"**スキル:** {eng['skills']}")
                        st.write(f"**経験年数:** {eng['experience_years']} 年")
                        st.markdown(f"**マッチング理由:**\n\n{result['reason']}")

        except Exception as e:
            err = str(e)
            if "does not exist" in err or "No collection" in err:
                st.error("インデックスが構築されていません。サイドバーから構築してください。")
            elif "AuthenticationError" in err or "api_key" in err.lower():
                st.error("APIキーが設定されていません。.env ファイルを確認してください。")
            else:
                st.error(f"API呼び出しに失敗しました。しばらく待ってから再試行してください。\n{err}")
