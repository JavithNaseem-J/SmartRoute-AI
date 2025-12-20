import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.pipeline.inference import HybridInferencePipeline
from src.cost.budget import BudgetManager
from src.cost.reporter import CostReporter
from src.cost.tracker import CostTracker

# Page config
st.set_page_config(
    page_title="Cost-Optimized RAG",
    page_icon="💰",
    layout="wide"
)

# Initialize
@st.cache_resource
def init_components():
    pipeline = HybridInferencePipeline()
    tracker = CostTracker()
    budget_manager = BudgetManager(tracker)
    reporter = CostReporter(tracker)
    return pipeline, tracker, budget_manager, reporter

pipeline, tracker, budget_manager, reporter = init_components()

# Title
st.title("💰 Cost-Optimized RAG System")
st.markdown("**HYBRID**: LangChain Retrieval + Custom Routing")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    strategy = st.selectbox(
        "Routing Strategy",
        ["cost_optimized", "quality_first", "balanced"],
        index=0
    )
    
    use_retrieval = st.checkbox("Use RAG", value=True)
    
    days_filter = st.selectbox(
        "Time Period",
        [1, 7, 30],
        format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
    )
    
    if st.button("🔄 Refresh"):
        st.cache_resource.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Query", 
    "📊 Cost Analytics", 
    "💰 Budget Status", 
    "📈 Performance"
])

# Tab 1: Query Interface
with tab1:
    st.header("Ask a Question")
    
    query = st.text_area(
        "Your Question",
        placeholder="What is machine learning?",
        height=100
    )
    
    if st.button("🚀 Ask", type="primary"):
        if query:
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
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model", result['model_used'])
                with col2:
                    st.metric("Complexity", result['complexity'])
                with col3:
                    st.metric("Cost", f"${result['cost']:.4f}")
                with col4:
                    st.metric("Latency", f"{result['latency']:.2f}s")
                
                if result['sources']:
                    with st.expander("📚 Sources"):
                        for source in result['sources']:
                            st.write(f"- {source}")
                
                with st.expander("🔄 Routing Info"):
                    st.json(result['routing_info'])
            else:
                st.error(f"❌ {result['error']}")

# Tab 2: Cost Analytics
with tab2:
    st.header(f"Cost Analytics - Last {days_filter} Day(s)")
    
    stats = tracker.get_statistics(days=days_filter)
    savings = tracker.calculate_savings(days=days_filter)
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cost by Model")
        if stats['by_model']:
            model_data = {m: d['cost'] for m, d in stats['by_model'].items()}
            fig = go.Figure(data=[go.Pie(
                labels=list(model_data.keys()),
                values=list(model_data.values()),
                hole=0.3
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data")
    
    with col2:
        st.subheader("Query Distribution")
        if stats['by_complexity']:
            comp_data = {c: d['count'] for c, d in stats['by_complexity'].items()}
            fig = go.Figure(data=[go.Bar(
                x=list(comp_data.keys()),
                y=list(comp_data.values()),
                marker_color=['#00CC96', '#FFA15A', '#EF553B']
            )])
            fig.update_layout(height=300, xaxis_title="Complexity", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data")

# Tab 3: Budget Status
with tab3:
    st.header("Budget Status")
    
    budget_status = budget_manager.get_budget_status()
    
    for period, data in budget_status.items():
        st.subheader(f"{period.capitalize()} Budget")
        
        progress = min(data['percentage'] / 100, 1.0)
        
        if data['alert']:
            st.warning(f"⚠️ {period.capitalize()} at {data['percentage']:.1f}%")
        
        st.progress(progress)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spent", f"${data['spent']:.4f}")
        with col2:
            st.metric("Limit", f"${data['limit']:.2f}")
        with col3:
            st.metric("Remaining", f"${data['remaining']:.4f}")
        
        st.markdown("---")

# Tab 4: Performance
with tab4:
    st.header("Performance Comparison")
    
    comparison = reporter.get_model_performance_comparison(days=days_filter)
    
    if comparison:
        st.subheader("Model Performance")
        
        df = pd.DataFrame(comparison).T
        df = df.round(4)
        st.dataframe(df, use_container_width=True)
        
        fig = go.Figure()
        for model, data in comparison.items():
            fig.add_trace(go.Scatter(
                x=[data['percentage_of_queries']],
                y=[data['percentage_of_cost']],
                mode='markers+text',
                name=model,
                text=[model],
                textposition="top center",
                marker=dict(size=15)
            ))
        
        fig.update_layout(
            title="Cost Efficiency",
            xaxis_title="% of Queries",
            yaxis_title="% of Total Cost",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")
