import json
import os
from pathlib import Path

import jwt
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SmartRoute Inference Gateway", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Generate a local JWT for the Streamlit UI to talk to the backend
jwt_secret = os.getenv(
    "SUPABASE_JWT_SECRET", "super-secret-jwt-token-with-at-least-32-characters-long"
)
ui_token = jwt.encode({"sub": "streamlit-ui", "role": "admin"}, jwt_secret, algorithm="HS256")
HEADERS = {"Authorization": f"Bearer {ui_token}"}


@st.cache_resource
def check_api_ready():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            return True
        return False
    except Exception:
        return False


ready = check_api_ready()

# Title
st.title("SmartRoute Inference Gateway")
st.header("Cost-Optimized LLM Inference with Smart Routing and RAG")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    if ready:
        st.success("API Connected")
    else:
        st.error("API Unavailable. Start `python -m api.main`")

    strategy = st.selectbox(
        "Routing Strategy", ["cost_optimized", "quality_first", "balanced"], index=0
    )

    use_retrieval = st.checkbox("Use RAG Retrieval", value=False)

    days_filter = st.selectbox(
        "Time Period",
        [1, 7, 30],
        format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
    )

    st.markdown("---")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Inference Console", "Cost Analytics", "Budget Status"])

# Tab 1: Query Interface
with tab1:
    docs_dir = Path("data/documents")
    existing_docs = list(docs_dir.glob("**/*.*")) if docs_dir.exists() else []
    doc_files = [f for f in existing_docs if f.suffix.lower() in [".pdf", ".txt", ".md"]]

    if use_retrieval:
        st.header("Enterprise Knowledge Base")

        if "docs_processed" not in st.session_state:
            st.session_state.docs_processed = len(doc_files) > 0

        if doc_files and st.session_state.docs_processed:
            st.success(f"Knowledge base active: {len(doc_files)} indexed document(s)")
            with st.expander("View Indexed Documents"):
                for doc in doc_files:
                    st.write(f"- {doc.name}")

            if st.checkbox("Index additional documents"):
                uploaded_files = st.file_uploader(
                    "Upload additional PDF, TXT, or MD files",
                    type=["pdf", "txt", "md"],
                    accept_multiple_files=True,
                    key="additional_upload",
                )

                if uploaded_files:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        process_btn = st.button("Process Documents", type="primary")

                    if process_btn:
                        with st.spinner("Uploading and indexing via API..."):
                            docs_dir.mkdir(parents=True, exist_ok=True)
                            processed_count = 0
                            for uploaded_file in uploaded_files:
                                file_path = docs_dir / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                processed_count += 1

                            try:
                                r = requests.post(f"{API_URL}/v1/index", headers=HEADERS)
                                r.raise_for_status()
                                st.session_state.docs_processed = True
                                st.success(f"✅ Indexed {processed_count} document(s) on backend!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Indexing Error: {e}")
        else:
            st.info("Upload source documents to enable retrieval-augmented generation (RAG).")
            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, or MD files to build knowledge base",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                key="initial_upload",
            )

            if uploaded_files:
                col1, col2 = st.columns([1, 4])
                with col1:
                    process_btn = st.button("Process Documents", type="primary")

                if process_btn:
                    with st.spinner("Uploading and indexing via API..."):
                        docs_dir.mkdir(parents=True, exist_ok=True)
                        processed_count = 0
                        for uploaded_file in uploaded_files:
                            file_path = docs_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            processed_count += 1

                        try:
                            r = requests.post(f"{API_URL}/v1/index", headers=HEADERS)
                            r.raise_for_status()
                            st.session_state.docs_processed = True
                            st.success(f"✅ Processed {processed_count} document(s) on backend!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Indexing Error: {e}")

        st.markdown("---")

    st.header("Inference Query")

    if use_retrieval and not st.session_state.get("docs_processed", False):
        st.warning(
            "Please index documents first to use RAG mode, or disable RAG in the configuration."
        )

    query = st.text_area(
        "Query Input",
        placeholder="Enter your query here..."
        if not use_retrieval
        else "Enter a query regarding the indexed documents...",
        height=100,
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        ask_button = st.button("Execute Query", type="primary", use_container_width=True)

    if ask_button and query and ready:
        st.markdown("### Output")
        response_placeholder = st.empty()

        full_text = ""
        metadata = {}
        final_result = {}

        try:
            with requests.post(
                f"{API_URL}/v1/query/stream",
                headers=HEADERS,
                json={"query": query, "strategy": strategy, "use_retrieval": use_retrieval},
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            data = json.loads(line_text[6:])
                            if data.get("type") == "metadata":
                                metadata = data.get("data", {})
                            elif data.get("type") == "chunk":
                                full_text += data.get("content", "")
                                response_placeholder.markdown(full_text + "▌")
                            elif data.get("type") == "done":
                                final_result = data.get("result", {})
                            elif data.get("type") == "error":
                                st.error(f"API Error: {data.get('content')}")

            response_placeholder.markdown(full_text)

            if final_result.get("success"):
                st.success(
                    f"Response generated successfully in {final_result.get('latency', 0):.2f}s."
                )

                sources = metadata.get("sources") or final_result.get("sources")
                if sources:
                    with st.expander("Source Attributions"):
                        for source in sources:
                            st.write(f"- {source}")

                routing_info = metadata.get("routing_info") or final_result.get("routing_info")
                if routing_info:
                    with st.expander("Detailed Routing Metrics"):
                        st.json(routing_info)
            else:
                st.error(f"❌ {final_result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"Connection failed: {e}")

# Tab 2: Cost Analytics
with tab2:
    st.header(f"Cost Analytics - Last {days_filter} Day(s)")

    if ready:
        try:
            stats = requests.get(
                f"{API_URL}/v1/stats", headers=HEADERS, params={"days": days_filter}
            ).json()
            savings = requests.get(
                f"{API_URL}/v1/savings", headers=HEADERS, params={"days": days_filter}
            ).json()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", stats.get("total_queries", 0))
            with col2:
                st.metric(
                    "Total Cost",
                    f"${stats.get('total_cost', 0):.4f}",
                    delta=f"-${savings.get('savings', 0):.4f}"
                    if savings.get("savings", 0) > 0
                    else None,
                    delta_color="inverse",
                )
            with col3:
                st.metric("Avg Cost/Query", f"${stats.get('avg_cost_per_query', 0):.4f}")
            with col4:
                st.metric(
                    "Savings",
                    f"{savings.get('percentage', 0):.1f}%",
                    delta=f"${savings.get('savings', 0):.4f}",
                )

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cost by Model")
                by_model = stats.get("by_model", {})
                if by_model:
                    model_data = pd.DataFrame(
                        [
                            {"Model": m, "Cost": d["cost"], "Count": d["count"]}
                            for m, d in by_model.items()
                        ]
                    )
                    fig = px.pie(
                        model_data,
                        values="Cost",
                        names="Model",
                        title="Cost Distribution",
                        hole=0.3,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available")

            with col2:
                st.subheader("Query Complexity Distribution")
                by_complexity = stats.get("by_complexity", {})
                if by_complexity:
                    comp_data = pd.DataFrame(
                        [{"Complexity": c, "Count": d["count"]} for c, d in by_complexity.items()]
                    )
                    fig = px.bar(
                        comp_data,
                        x="Complexity",
                        y="Count",
                        title="Queries by Complexity",
                        color="Complexity",
                        color_discrete_map={
                            "simple": "#00CC96",
                            "medium": "#FFA15A",
                            "complex": "#EF553B",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available")

            st.subheader("Model Performance Details")
            if by_model:
                total_q = stats.get("total_queries", 1) or 1
                model_df = pd.DataFrame(
                    [
                        {
                            "Model": m,
                            "Queries": d["count"],
                            "Total Cost": f"${d['cost']:.4f}",
                            "Avg Cost": f"${d.get('avg_cost', d['cost']/d['count'] if d['count']>0 else 0):.4f}",
                            "Percentage": f"{(d['count']/total_q*100):.1f}%",
                        }
                        for m, d in by_model.items()
                    ]
                )
                st.dataframe(model_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")

# Tab 3: Budget Status
with tab3:
    st.header("Budget Status")

    if ready:
        try:
            budget_status = requests.get(f"{API_URL}/v1/budget", headers=HEADERS).json()
            for period, data in budget_status.items():
                if period in ["alert_threshold", "timestamp"]:
                    continue
                st.subheader(f"{period.capitalize()} Budget")
                progress = min(data.get("percentage", 0) / 100, 1.0)
                if data.get("alert"):
                    st.warning(
                        f"⚠️ {period.capitalize()} budget at {data.get('percentage', 0):.1f}%"
                    )
                st.progress(progress)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Spent", f"${data.get('spent', 0):.4f}")
                with col2:
                    st.metric("Limit", f"${data.get('limit', 0):.2f}")
                with col3:
                    st.metric("Remaining", f"${data.get('remaining', 0):.4f}")
                st.markdown("---")
        except Exception as e:
            st.error(f"Failed to load budget: {e}")
