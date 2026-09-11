import os
import re
import time
import streamlit as st
from pathlib import Path
from io import BytesIO

from app.ingestion.pipeline import ingest_pdf
from app.embeddings.vectorstore import create_vectorstore
from app.retrieval.dense_retriever import DenseVectorRetriever
from app.retrieval.bm25_retriever import BM25LexicalRetriever
from app.retrieval.hybrid_rrf import HybridRRFRetriever
from app.retrieval.multiquery_retriever import MultiQueryRetriever
from app.retrieval.hyde_retriever import HyDERetriever
from app.retrieval.rerank_retriever import CrossEncoderRerankRetriever
from app.retrieval.parent_retriever import ParentDocumentRetriever
from app.retrieval.router_retriever import AdaptiveRouterRetriever
from app.retrieval.comparator import RetrievalComparator
from app.chains.rag_chain import create_rag_chain
from app.chains.comparison_chain import ComparisonRAGChain
from app.evaluation.benchmark import run_benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepPDF - Advanced Retrieval RAG & Benchmarks",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS - Modern Premium Theme styled like TransYT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --primary-dark: #090d16;
        --primary-light: #f8fafc;
        --accent-indigo: #4338ca;
        --accent-cyan: #0284c7;
        --accent-teal: #0d9488;
        --border-light: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-light: #ffffff;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border-light);
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid var(--border-light);
    }

    .logo-container img {
        height: 44px;
        width: auto;
    }

    .logo-text {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #312e81 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1.2rem;
    }

    .status-ready {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }

    .status-idle {
        background: #f1f5f9;
        color: #64748b;
        border: 1px solid #cbd5e1;
    }

    .status-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    .status-ready .status-dot {
        background: #10b981;
    }

    .status-idle .status-dot {
        background: #94a3b8;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #64748b;
        margin: 1.2rem 0 0.6rem;
    }

    .stats-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 12px 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #4338ca;
        line-height: 1.1;
    }

    .stat-label {
        font-size: 0.65rem;
        color: #64748b;
        margin-top: 4px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .retrieval-pill {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .welcome-box {
        background: #ffffff;
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        text-align: center;
        max-width: 760px;
        margin: 2rem auto;
    }

    .welcome-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e1b4b;
        margin-bottom: 0.8rem;
    }

    .welcome-subtitle {
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.8rem;
    }

    .tech-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 1rem;
    }

    .tech-badge {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_logo_base64():
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        import base64
        with open(logo_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    return None


logo_data = get_logo_base64()


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "is_ready": False,
    "pdf_name": "",
    "total_pages": 0,
    "total_chunks": 0,
    "retriever": None,
    "strategies": {},
    "comparator": None,
    "comp_chain": None,
    "active_strategy_key": "hybrid",
    "parent_docstore": None,
}

for k_key, v_val in defaults.items():
    if k_key not in st.session_state:
        st.session_state[k_key] = v_val


def init_strategies(chunks, child_chunks, parent_docstore, persist_dir="data/vectorstore"):
    vs = create_vectorstore(chunks, persist_directory=persist_dir, recreate=True)
    child_vs = create_vectorstore(child_chunks, persist_directory=f"{persist_dir}_child", recreate=True)

    dense = DenseVectorRetriever(vs)
    bm25 = BM25LexicalRetriever(chunks)
    hybrid = HybridRRFRetriever(vs, chunks)
    mq = MultiQueryRetriever(vs)
    hyde = HyDERetriever(vs)
    rerank = CrossEncoderRerankRetriever(vs)
    parent_ret = ParentDocumentRetriever(child_vs, parent_docstore)

    strats = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "multiquery": mq,
        "hyde": hyde,
        "rerank": rerank,
        "parent": parent_ret,
    }

    router = AdaptiveRouterRetriever(strats)
    strats["router"] = router

    comparator = RetrievalComparator(strats)
    comp_chain = ComparisonRAGChain(comparator)

    return strats, comparator, comp_chain


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if logo_data:
        st.markdown(
            f"""
            <div class="logo-container">
                <img src="{logo_data}" alt="DeepPDF">
                <div class="logo-text">DeepPDF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="logo-text">DeepPDF</div>', unsafe_allow_html=True)

    status_class = "status-ready" if st.session_state.is_ready else "status-idle"
    status_text = "Knowledge Base Active" if st.session_state.is_ready else "Awaiting PDF Document"
    st.markdown(
        f'<div class="status-badge {status_class}"><span class="status-dot"></span>{status_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Upload PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    load_sample = col1.button("Use Sample PDF", use_container_width=True)
    process_btn = col2.button("Process PDF", type="primary", use_container_width=True)

    if load_sample:
        sample_path = Path("data/sample_docs/sample_research_paper.pdf")
        if sample_path.exists():
            with st.status("Ingesting and Indexing Sample PDF...", expanded=True) as status_ui:
                st.write("Extracting pages & text...")
                chunks, _ = ingest_pdf(sample_path)
                child_chunks, parent_docstore = ingest_pdf(sample_path, hierarchical=True)
                st.write("Initializing 7 Advanced Retrieval Engines...")
                strats, comparator, comp_chain = init_strategies(chunks, child_chunks, parent_docstore)
                st.session_state.update(
                    is_ready=True,
                    pdf_name="sample_research_paper.pdf",
                    total_pages=3,
                    total_chunks=len(chunks),
                    strategies=strats,
                    comparator=comparator,
                    comp_chain=comp_chain,
                    parent_docstore=parent_docstore,
                    messages=[],
                )
                status_ui.update(label="Ready - Sample PDF Loaded", state="complete")
                st.rerun()

    if process_btn and uploaded_file:
        with st.status(f"Ingesting {uploaded_file.name}...", expanded=True) as status_ui:
            file_bytes = uploaded_file.read()
            st.write("Parsing document hierarchy...")
            chunks, _ = ingest_pdf(file_bytes, file_name=uploaded_file.name)
            child_chunks, parent_docstore = ingest_pdf(file_bytes, file_name=uploaded_file.name, hierarchical=True)
            st.write("Building ChromaDB vector indices & BM25 sparse index...")
            strats, comparator, comp_chain = init_strategies(chunks, child_chunks, parent_docstore)
            st.session_state.update(
                is_ready=True,
                pdf_name=uploaded_file.name,
                total_pages=max(d.metadata.get("page", 1) for d in chunks),
                total_chunks=len(chunks),
                strategies=strats,
                comparator=comparator,
                comp_chain=comp_chain,
                parent_docstore=parent_docstore,
                messages=[],
            )
            status_ui.update(label=f"Ready - {uploaded_file.name} Indexed", state="complete")
            st.rerun()

    if st.session_state.is_ready:
        st.markdown('<div class="section-label">Retrieval Strategy</div>', unsafe_allow_html=True)
        strategy_options = {
            "hybrid": "🔥 Hybrid Search (BM25 + Dense RRF)",
            "dense": "🔹 Dense Vector (Baseline)",
            "bm25": "🔸 BM25 Lexical Keyword",
            "multiquery": "🔀 Multi-Query Expansion",
            "hyde": "💡 HyDE (Hypothetical Embeddings)",
            "rerank": "🎯 Cross-Encoder Re-ranking",
            "parent": "🌳 Parent-Document (Hierarchical)",
            "router": "⚡ Adaptive Query Router",
            "compare_all": "📊 Compare All Strategies Side-by-Side",
        }
        chosen_key = st.selectbox(
            "Select Strategy",
            options=list(strategy_options.keys()),
            format_func=lambda k: strategy_options[k],
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.active_strategy_key = chosen_key

        st.markdown('<div class="section-label">Document Stats</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{st.session_state.total_pages}</div>
                    <div class="stat-label">Pages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{st.session_state.total_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.messages:
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.is_ready:
    st.markdown(
        f"""
        <div class="welcome-box">
            <h1 class="welcome-title">DeepPDF</h1>
            <p class="welcome-subtitle">
                Advanced Retrieval-Augmented Generation (RAG) system for technical PDFs.
                Benchmark and compare 7 state-of-the-art retrieval strategies with quantitative precision metrics.
            </p>
            <div class="tech-badges">
                <span class="tech-badge">🔥 Hybrid Search (BM25 + RRF)</span>
                <span class="tech-badge">🎯 Cross-Encoder Re-Ranking</span>
                <span class="tech-badge">🔀 Multi-Query Expansion</span>
                <span class="tech-badge">💡 HyDE Embeddings</span>
                <span class="tech-badge">🌳 Parent-Document Hierarchical</span>
                <span class="tech-badge">⚡ Adaptive Intent Router</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    tab_chat, tab_compare, tab_benchmark = st.tabs(["💬 Question & Answer", "🔬 Strategy Comparator Studio", "📊 Quantitative Benchmarks"])

    # TAB 1: Chat Q&A
    with tab_chat:
        strat_key = st.session_state.active_strategy_key
        st.markdown(f'<div class="retrieval-pill">Active Mode: {strat_key.upper()}</div>', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Context ({len(sources)} Chunks)"):
                        for idx, s in enumerate(sources, 1):
                            st.markdown(f"**Chunk {idx} — Page {s.get('page', '?')}**")
                            st.caption(s.get("text", "")[:600] + "...")
                            if idx < len(sources):
                                st.divider()

        if prompt := st.chat_input("Ask a question about the PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving with active strategy and generating answer..."):
                    if strat_key == "compare_all":
                        st.info("Switched to Strategy Comparison Studio. Please see the 'Strategy Comparator Studio' tab.")
                        ans = "Executed multi-strategy comparison. View the side-by-side results in the 'Strategy Comparator Studio' tab!"
                        sources_list = []
                    else:
                        strategy = st.session_state.strategies.get(strat_key) or st.session_state.strategies["hybrid"]
                        docs = strategy.retrieve(prompt, k=4)
                        context_str = "\n\n".join(f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs)
                        chain = create_rag_chain()
                        ans = chain.invoke({"context": context_str, "question": prompt})
                        sources_list = [{"page": d.metadata.get("page", 1), "text": d.page_content} for d in docs]

                st.markdown(ans)
                if sources_list:
                    with st.expander(f"📚 Retrieved Context ({len(sources_list)} Chunks)"):
                        for idx, s in enumerate(sources_list, 1):
                            st.markdown(f"**Chunk {idx} — Page {s.get('page', '?')}**")
                            st.caption(s.get("text", "")[:600] + "...")
                            if idx < len(sources_list):
                                st.divider()

            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": sources_list})

    # TAB 2: Comparator Studio
    with tab_compare:
        st.subheader("🔬 Side-by-Side Retrieval Strategy Comparison")
        st.write("Run your query concurrently across all 7 retrieval techniques to contrast latency, retrieved chunk overlap, and generated answers.")

        comp_q = st.text_input("Comparison Query:", value="What are the key differences between dense and hybrid retrieval?")
        if st.button("Run Multi-Strategy Comparison", type="primary"):
            with st.spinner("Executing all retrieval strategies in parallel..."):
                comp_results = st.session_state.comp_chain.run_comparison(comp_q, k=3)
                
                cols = st.columns(3)
                strat_list = list(comp_results["strategies"].items())
                for i, (k_name, s_data) in enumerate(strat_list):
                    with cols[i % 3]:
                        st.markdown(f"### {s_data['strategy_name']}")
                        st.metric("Latency", f"{s_data['latency_ms']} ms", delta=f"{s_data['doc_count']} docs")
                        st.markdown(f"**Answer:**\n{s_data.get('answer', 'N/A')[:400]}...")
                        with st.expander("Top Retrieved Passage"):
                            st.caption(s_data["preview"])
                        st.divider()

    # TAB 3: Quantitative Benchmarks
    with tab_benchmark:
        st.subheader("📊 Automated Quantitative Retrieval Benchmarking")
        st.write("Comprehensive benchmark comparing Precision@K, Mean Reciprocal Rank (MRR@K), Latency, and Lexical Diversity across 7 retrieval pipelines.")

        if st.button("Run Full Benchmark Suite"):
            with st.spinner("Benchmarking all retrieval strategies..."):
                b_results = run_benchmark(st.session_state.strategies, k=4)
                
                import pandas as pd
                df = pd.DataFrame.from_dict(b_results, orient="index")
                df = df.rename(columns={
                    "name": "Strategy",
                    "avg_latency_ms": "Avg Latency (ms)",
                    "precision_at_k": "Precision@K",
                    "mrr": "MRR@K",
                    "diversity": "Lexical Diversity",
                })
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Strategy")["Avg Latency (ms)"])
