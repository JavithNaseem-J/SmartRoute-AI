import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.inference import InferencePipeline
from src.retrieval.indexer import DocumentIndexer

st.set_page_config(
    page_title="SmartRoute-AI",
    page_icon="💰",
    layout="wide"
)

# Initialize
@st.cache_resource
def init_pipeline():
    return InferencePipeline()

@st.cache_resource
def init_indexer():
    return DocumentIndexer()

try:
    pipeline = init_pipeline()
    indexer = init_indexer()
    ready = True
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    ready = False


# Title
st.title("💰SmartRoute-AI")
st.header("Cost-Optimized LLM Inference with Smart Routing and RAG")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if ready:
        st.success("System Ready")
    else:
        st.error("System Error")
    
    strategy = st.selectbox(
        "Routing Strategy",
        ["cost_optimized", "quality_first", "balanced"],
        index=0
    )
    
    use_retrieval = st.checkbox("Use RAG Retrieval", value=False)
    
    days_filter = st.selectbox(
        "Time Period",
        [1, 7, 30],
        format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
    )
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Query Interface",
    "📊 Cost Analytics",
    "💰 Budget Status"])

# Tab 1: Query Interface
with tab1:
    # Check if documents are already uploaded
    docs_dir = Path("data/documents")
    existing_docs = list(docs_dir.glob("**/*.*")) if docs_dir.exists() else []
    doc_files = [f for f in existing_docs if f.suffix.lower() in ['.pdf', '.txt', '.md']]
    
    # RAG Mode Section - Only show if RAG is enabled
    if use_retrieval:
        st.header("📄 Document Knowledge Base")
        
        # Initialize session state for tracking upload status
        if 'docs_processed' not in st.session_state:
            st.session_state.docs_processed = len(doc_files) > 0
        
        # Show existing documents if any
        if doc_files and st.session_state.docs_processed:
            st.success(f"📚 Knowledge base has {len(doc_files)} document(s)")
            
            with st.expander("📋 View Uploaded Documents"):
                for doc in doc_files:
                    st.write(f"- {doc.name}")
            
            # Option to add more documents
            if st.checkbox("➕ Add more documents"):
                uploaded_files = st.file_uploader(
                    "Upload additional PDF, TXT, or MD files",
                    type=['pdf', 'txt', 'md'],
                    accept_multiple_files=True,
                    key="additional_upload"
                )
                
                if uploaded_files:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        process_btn = st.button("📥 Process", type="primary")
                    
                    if process_btn:
                        with st.spinner("Processing documents..."):
                            docs_dir.mkdir(parents=True, exist_ok=True)
                            processed_count = 0
                            for uploaded_file in uploaded_files:
                                file_path = docs_dir / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                processed_count += 1
                            
                            try:
                                indexer.index_directory(docs_dir)
                                pipeline.retriever.reload()
                                st.session_state.docs_processed = True
                                st.success(f"✅ Added {processed_count} document(s)!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
        else:
            # No documents yet - show upload interface
            st.info("📤 Upload documents to enable RAG-based question answering")
            
            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, or MD files to build knowledge base",
                type=['pdf', 'txt', 'md'],
                accept_multiple_files=True,
                key="initial_upload"
            )
            
            if uploaded_files:
                col1, col2 = st.columns([1, 4])
                with col1:
                    process_btn = st.button("📥 Process Documents", type="primary")
                
                if process_btn:
                    with st.spinner("Processing documents..."):
                        docs_dir.mkdir(parents=True, exist_ok=True)
                        processed_count = 0
                        for uploaded_file in uploaded_files:
                            file_path = docs_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            processed_count += 1
                        
                        try:
                            indexer.index_directory(docs_dir)
                            pipeline.retriever.reload()
                            st.session_state.docs_processed = True
                            stats = indexer.get_stats()
                            st.success(f"✅ Processed {processed_count} document(s)!")
                            st.info(f"📊 Vector store has {stats.get('vector_store', {}).get('document_count', 0)} chunks")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        st.markdown("---")
    
    # Query Section
    st.header("Ask a Question❓")
    
    if use_retrieval and not st.session_state.get('docs_processed', False):
        st.warning("⚠️ Please upload documents first to use RAG mode, or disable RAG in sidebar.")
    
    query = st.text_area(
        "Your Question",
        placeholder="Ask your question here..." if not use_retrieval else "Ask questions about your uploaded documents...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    if ask_button and query and ready:
        with st.spinner("Processing..."):
            result = pipeline.run(
                query=query,
                strategy=strategy,
                use_retrieval=use_retrieval
            )
        
        if result['success']:
            st.success("✓ Answer generated")
            
            st.markdown("### Answer")
            st.write(result['answer'])
            
            
            if result.get('sources'):
                with st.expander("📚 Sources"):
                    for source in result['sources']:
                        st.write(f"- {source}")
            
            if result.get('routing_info'):
                with st.expander("🔄 Full Routing Info"):
                    st.json(result['routing_info'])
        else:
            st.error(f"❌ {result['error']}")

# Tab 2: Cost Analytics
with tab2:
    st.header(f"Cost Analytics - Last {days_filter} Day(s)")
    
    if ready:
        stats = pipeline.tracker.get_statistics(days=days_filter)
        savings = pipeline.tracker.calculate_savings(days=days_filter)
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", stats['total_queries'])
        
        with col2:
            st.metric(
                "Total Cost",
                f"${stats['total_cost']:.4f}",
                delta=f"-${savings['savings']:.4f}" if savings['savings'] > 0 else None,
                delta_color="inverse"
            )
        
        with col3:
            st.metric("Avg Cost/Query", f"${stats['avg_cost_per_query']:.4f}")
        
        with col4:
            st.metric(
                "Savings",
                f"{savings['percentage']:.1f}%",
                delta=f"${savings['savings']:.4f}"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost by Model")
            if stats['by_model']:
                model_data = pd.DataFrame([
                    {'Model': m, 'Cost': d['cost'], 'Count': d['count']}
                    for m, d in stats['by_model'].items()
                ])
                
                fig = px.pie(
                    model_data,
                    values='Cost',
                    names='Model',
                    title='Cost Distribution',
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available")
        
        with col2:
            st.subheader("Query Complexity Distribution")
            if stats['by_complexity']:
                comp_data = pd.DataFrame([
                    {'Complexity': c, 'Count': d['count']}
                    for c, d in stats['by_complexity'].items()
                ])
                
                fig = px.bar(
                    comp_data,
                    x='Complexity',
                    y='Count',
                    title='Queries by Complexity',
                    color='Complexity',
                    color_discrete_map={
                        'simple': '#00CC96',
                        'medium': '#FFA15A',
                        'complex': '#EF553B'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available")
        
        # Model performance table
        st.subheader("Model Performance Details")
        if stats['by_model']:
            model_df = pd.DataFrame([
                {
                    'Model': m,
                    'Queries': d['count'],
                    'Total Cost': f"${d['cost']:.4f}",
                    'Avg Cost': f"${d['avg_cost']:.4f}",
                    'Percentage': f"{(d['count']/stats['total_queries']*100):.1f}%"
                }
                for m, d in stats['by_model'].items()
            ])
            st.dataframe(model_df, use_container_width=True)

# Tab 3: Budget Status
with tab3:
    st.header("Budget Status")
    
    if ready:
        budget_status = pipeline.budget_manager.get_budget_status()
        
        for period, data in budget_status.items():
            if period in ['alert_threshold', 'timestamp']:
                continue
            
            st.subheader(f"{period.capitalize()} Budget")
            
            progress = min(data['percentage'] / 100, 1.0)
            
            if data['alert']:
                st.warning(f"⚠️ {period.capitalize()} budget at {data['percentage']:.1f}%")
            
            st.progress(progress)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Spent", f"${data['spent']:.4f}")
            with col2:
                st.metric("Limit", f"${data['limit']:.2f}")
            with col3:
                st.metric("Remaining", f"${data['remaining']:.4f}")
            
            st.markdown("---")