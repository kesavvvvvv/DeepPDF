<p align="center">
  <img src="assets/logo.png" alt="DeepPDF Logo" width="28%">
</p>
<h1 align="center">DeepPDF</h1>

**Transform complex PDF documents into an intelligent, high-precision knowledge base with comparative multi-strategy retrieval.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-green)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLM-purple)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE.md)

---

## 🚀 Overview

**DeepPDF** is an enterprise-grade Document Intelligence and Question-Answering platform built to overcome the fundamental retrieval failure modes of naive RAG. While traditional vector search suffers from vocabulary mismatch, lost-in-the-middle context loss, and poor keyword fidelity on technical manuals, DeepPDF implements and compares **7 Advanced Retrieval Strategies** within a unified benchmark architecture.

---

## ⚡ Key Features

- **📑 Multi-Engine PDF Ingestion** - Fast, high-fidelity parsing of complex technical PDFs with precise page mapping and section metadata.
- **🔬 7 Advanced Retrieval Techniques**:
  1. **Standard Dense Vector Retrieval** (Baseline cosine similarity search via ChromaDB)
  2. **BM25 Lexical Keyword Search** (Okapi BM25 scoring for exact keyword, formula, and acronym matches)
  3. **Hybrid Search with Reciprocal Rank Fusion (RRF)** (Mathematically fuses sparse + dense retrieval ranks)
  4. **Multi-Query Expansion & Decomposition** (LLM query reformulation from multiple diverse perspectives)
  5. **Hypothetical Document Embeddings (HyDE)** (Synthesizes hypothetical answer documents to bridge embedding gaps)
  6. **Two-Stage Cross-Encoder Re-Ranking** (High-recall candidate pool + deep cross-attention precision scoring)
  7. **Parent-Document / Hierarchical Chunking** (Fine-grained child vectors for search + expansive parent context for generation)
- **⚡ Adaptive Intent Router** - Dynamically analyzes question semantics and automatically dispatches queries to the optimal retrieval pipeline.
- **📊 Real-Time Retrieval Comparator Studio** - Compare results, latencies, chunk overlap, and generated answers across all 7 retrieval techniques simultaneously.
- **📈 Quantitative Benchmark Suite** - Automated evaluation calculating **Precision@K**, **Mean Reciprocal Rank (MRR@K)**, **Lexical Diversity**, and **Latency (ms)**.
- **🎨 Premium Modern UI** - Streamlit interface with drag-and-drop PDF upload, page inspection, statistics cards, and conversational source attribution.

---

## 🛠️ Tech Stack

### **Core Framework**
- **[Streamlit](https://streamlit.io)** - Modern interactive web application UI
- **[LangChain](https://langchain.com)** - LLM orchestration and LCEL execution chains

### **AI & Language Models**
- **[Groq LLaMA 3.3](https://groq.com)** - High-speed LLM inference (`llama-3.3-70b-versatile`, temperature = 0)
- **[Google Gemini](https://ai.google.dev/)** & **[OpenAI](https://openai.com/)** - Multi-provider fallback support
- **[HuggingFace Embeddings](https://huggingface.co)** - `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **[Cross-Encoder](https://sbert.net)** - `cross-encoder/ms-marco-MiniLM-L-6-v2` for two-stage re-ranking

### **Vector Database & Lexical Search**
- **[ChromaDB](https://www.trychroma.com)** - Persistent HNSW vector indexing
- **[Rank-BM25](https://github.com/dorianbrown/rank_bm25)** - Okapi BM25 sparse keyword retriever

### **Document Ingestion**
- **[PyPDF](https://pypdf.readthedocs.io/)** - High-fidelity PDF document parser and page separator

---

## 📋 Architecture

```
DeepPDF
├── app/
│   ├── ingestion/             # Document loading & splitting
│   │   ├── pdf_loader.py      # PDF text & metadata extractor
│   │   ├── text_splitter.py   # Recursive & hierarchical chunking
│   │   └── pipeline.py        # Ingestion orchestrator
│   ├── embeddings/            # Vector representations
│   │   ├── embedder.py        # HuggingFace / GenAI embedding factory
│   │   └── vectorstore.py     # ChromaDB persistence & collections
│   ├── retrieval/             # 7 Advanced Retrieval Engines
│   │   ├── base.py            # Abstract Base Strategy interface
│   │   ├── dense_retriever.py # Cosine dense vector search
│   │   ├── bm25_retriever.py  # BM25 lexical keyword search
│   │   ├── hybrid_rrf.py      # BM25 + Dense Reciprocal Rank Fusion
│   │   ├── multiquery_retriever.py # Query expansion & decomposition
│   │   ├── hyde_retriever.py  # Hypothetical Document Embeddings
│   │   ├── rerank_retriever.py# Two-stage Cross-Encoder re-ranker
│   │   ├── parent_retriever.py# Hierarchical child-parent retriever
│   │   ├── router_retriever.py# Adaptive self-routing retrieval
│   │   └── comparator.py      # Multi-strategy comparison engine
│   ├── chains/                # LangChain composition
│   │   ├── prompts.py         # Grounded system prompts & templates
│   │   ├── rag_chain.py       # LCEL execution chain
│   │   └── comparison_chain.py# Multi-pipeline concurrent execution
│   ├── evaluation/            # Evaluation & metrics
│   │   ├── metrics.py         # Precision@K, MRR@K, Overlap, Diversity
│   │   └── benchmark.py       # Benchmark runner
│   └── models/                # LLM initialization
│       └── llm.py             # Multi-provider model factory (Groq/Gemini/OpenAI)
├── assets/                    # Static assets & branding
│   └── logo.png               # DeepPDF logo
├── data/
│   ├── sample_docs/           # Sample PDF research papers
│   └── vectorstore/           # ChromaDB persistent store
├── tests/                     # Unit tests & verification
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retrieval_dense.py
│   ├── test_retrieval_hybrid.py
│   ├── test_retrieval_multiquery.py
│   ├── test_retrieval_hyde.py
│   ├── test_retrieval_rerank.py
│   ├── test_retrieval_parent.py
│   ├── test_comparator.py
│   ├── test_rag_e2e.py
│   ├── test_retrieval.py
│   └── test_rag.py
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
└── README.md                  # Documentation
```

---

## 🧠 Deep Dive: Advanced Retrieval Techniques

```
                               ┌─────────────────────────────┐
                               │       User Question         │
                               └──────────────┬──────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
           ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
           │   1. Dense Vector    ││   2. BM25 Lexical    ││ 3. Multi-Query / HyDE│
           │ (Cosine / ChromaDB)  ││   (Okapi Term Freq)  ││  (LLM Formulation)   │
           └──────────┬───────────┘└──────────┬───────────┘└──────────┬───────────┘
                      │                       │                       │
                      └───────────────────────┼───────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  4. Reciprocal Rank Fusion  │
                               │  RRF(d) = Σ 1 / (60 + r(d)) │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  5. Cross-Encoder Re-Ranker │
                               │  (Top-k High Precision)     │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ 6. Parent-Document Context  │
                               │ (Expand child -> full page) │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │      Groq LLaMA 3.3 LLM     │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ Grounded Answer + Citations │
                               └─────────────────────────────┘
```

### **1. Hybrid Search with Reciprocal Rank Fusion (RRF)**
Merges sparse lexical rankings (BM25) and dense semantic vector rankings by computing:
$$\text{RRF Score}(d) = \sum_{r \in \{\text{BM25}, \text{Dense}\}} \frac{w_r}{k_{\text{rrf}} + \text{rank}_r(d)}$$
This eliminates score distribution calibration issues and combines exact keyword precision (acronyms, IDs) with deep conceptual retrieval.

### **2. Two-Stage Cross-Encoder Re-Ranking**
Bi-encoders independently embed query and passages, missing fine-grained token-level cross-attention. DeepPDF retrieves a wide candidate pool ($N=20$) and uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to perform all-to-all query-document attention, re-ordering candidate passages by true semantic relevance.

### **3. Hypothetical Document Embeddings (HyDE)**
Standard user questions are short and phrased as interrogatives, creating an embedding space divergence against declarative PDF passages. HyDE generates a synthetic hypothetical answer using an LLM, embedding that passage to match real passages in latent vector space.

### **4. Parent-Document / Hierarchical Chunking**
Small chunks (e.g. 250 characters) optimize vector search granularity, but lose surrounding narrative context. DeepPDF indexes child chunks while preserving links to their parent chunk (1200 characters), returning rich, unbroken context to the LLM.

---

## 📊 Benchmark & Performance Comparison

Empirical evaluation conducted on the comprehensive **Neuralink N1 Brain-Computer Interface Technical Whitepaper** (6 densely packed pages spanning electrode arrays, CMOS ASICs, R1 robotic microsurgery, inductive power transfer, and PRIME human clinical trial results):

| Retrieval Strategy | Avg Latency (ms) | Precision@4 | MRR@4 | Lexical Diversity | Primary Strengths & Best Suited For |
|---|---|---|---|---|---|
| **Dense Vector (Baseline)** | 18.21 ms | 0.750 | **1.000** | 0.687 | Standard conceptual and semantic questions |
| **BM25 Lexical Search** | **0.35 ms** | 0.812 | 0.875 | 0.715 | Exact numerical specs (e.g. `1024 electrodes`, `6.78 MHz`) |
| **Hybrid Search (BM25 + RRF)** | 13.72 ms | **0.875** | 0.875 | 0.703 | **General production RAG — balances keyword & semantic accuracy (Recommended)** |
| **Multi-Query Expansion** | 28.81 ms | 0.750 | **1.000** | 0.687 | Ambiguous or multifaceted research questions |
| **HyDE (Hypothetical Embeddings)**| 12.55 ms | 0.750 | **1.000** | 0.687 | Abstract / inferential questions bridging vocabulary gaps |
| **Cross-Encoder Re-Ranking** | 17.83 ms | 0.688 | **1.000** | **0.716** | Precision-critical surgical / regulatory queries with candidate scoring |
| **Parent-Document (Hierarchical)**| 14.09 ms | 0.812 | **1.000** | 0.690 | Long structured technical papers requiring complete contextual sections |

---

### 🔬 Technical Benchmark Dataset: Neuralink N1 Whitepaper
The benchmark suite evaluates dense retrieval across nine technical domains:
1. **Electrode Array Architecture**: 1,024 electrodes across 64 polyimide threads (30 μm pitch, 4–6 μm width).
2. **Custom ASIC & Bio-Signal Processing**: 200× on-chip amplification, 20 kHz sampling, 0.75 μV RMS noise floor.
3. **R1 Surgical Robotics**: Micron-scale optical coherence tomography (OCT) and automated vascular avoidance.
4. **Wireless Telemetry & Power Transfer**: 2.4 GHz Bluetooth Low Energy (BLE), AES-128-GCM encryption, 6.78 MHz ISM inductive charging.
5. **Hermetic Packaging & Biocompatibility**: Grade 23 Ti-6Al-4V-ELI titanium alloy enclosure with sapphire optical windows.
6. **PRIME Human Clinical Trial Outcomes**: First-in-human tetraplegia cursor control achieving peak information transfer rates of 10.2 bits per second (BPS).


---

## 📦 Installation & Setup

### **Prerequisites**
- Python 3.12 or higher
- `pip` or `uv` package manager
- Groq API Key (Free tier available at [console.groq.com](https://console.groq.com))

### **Step 1: Clone Repository**
```bash
git clone https://github.com/kesavvvvvv/DeepPDF.git
cd DeepPDF
```

### **Step 2: Create & Activate Virtual Environment**
```bash
# Using venv
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
# Using uv (recommended - ultra fast):
uv pip install -r requirements.txt

# Or using pip:
pip install -r requirements.txt
```

### **Step 4: Configure Environment Variables**
Create a `.env` file in the project root:
```env
# Groq Cloud API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🎯 Running the Application

### **Start the Streamlit Web UI**
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**.

### **Running CLI Tests & Benchmarks**
```bash
# Run automated pytest suite
python -m pytest tests/

# Test PDF Ingestion
python tests/test_ingestion.py

# Test Multi-Strategy Retrieval Comparator
python tests/test_comparator.py

# Run interactive RAG test
python tests/test_rag.py
```

---

## 💡 Usage Guide

1. **Upload or Select PDF**:
   - Drag & drop your PDF in the sidebar, or click **"Use Sample PDF"** to test immediately.
2. **Select Retrieval Technique**:
   - Choose from 7 specialized retrieval strategies or select **"Compare All Strategies Side-by-Side"**.
3. **Ask Natural Language Questions**:
   - Ask complex questions and receive structured, grounded answers with page-level citations.
4. **Inspect Context & Benchmarks**:
   - Open the **"Strategy Comparator Studio"** tab to see cross-strategy performance side-by-side.

---

## 📁 Project Structure Details

- **`app/ingestion/`**: Multi-engine PDF parser and recursive/hierarchical document chunkers.
- **`app/embeddings/`**: HuggingFace sentence transformers (`all-MiniLM-L6-v2`) & ChromaDB vector store.
- **`app/retrieval/`**: 7 advanced retrieval strategies with RRF, Cross-Encoders, HyDE, Multi-Query, and Auto-Routing.
- **`app/evaluation/`**: Metric computation (MRR, Precision@K, Jaccard Overlap, Diversity) and benchmark suite.
- **`app/chains/`**: LangChain LCEL RAG execution chains and grounded prompts.
- **`app/models/`**: LLM provider factory with Groq LLaMA 3.3, Google Gemini, and OpenAI.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE.md](LICENSE.md) file for details.

---

## 🙏 Acknowledgments

- **Groq** for high-speed LLaMA 3.3 inference
- **LangChain** for robust LCEL composability
- **ChromaDB** for efficient vector indexing
- **HuggingFace** for state-of-the-art embedding and cross-encoder models
- **Streamlit** for the interactive web framework
