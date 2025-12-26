import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.inference import InferencePipeline

# Page config
st.set_page_config(
    page_title="Cost-Optimized RAG",
    page_icon="💰",
    layout="wide"
)

# Initialize
@st.cache_resource
def init_pipeline():
    return InferencePipeline()

try:
    pipeline = init_pipeline()
    ready = True
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    ready = False
    pipeline = None

# Title
st.title("💰 Cost-Optimized RAG System")
st.markdown("**Smart routing for 70% cost savings**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if ready:
        st.success("✓ System Ready")
    else:
        st.error("✗ System Error")
    
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
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Query Interface",
    "📊 Cost Analytics",
    "💰 Budget Status",
    "📈 Model Performance"
])

# Tab 1: Query Interface
with tab1:
    st.header("Ask a Question")
    
    query = st.text_area(
        "Your Question",
        placeholder="What is machine learning?",
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
            
            with st.expander("🔄 Routing Details"):
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

# Tab 4: Model Performance
with tab4:
    st.header("Model Performance Comparison")
    
    if ready:
        stats = pipeline.tracker.get_statistics(days=days_filter)
        
        if stats['by_model']:
            # Create comparison dataframe
            comparison_data = []
            for model_id, model_stats in stats['by_model'].items():
                comparison_data.append({
                    'Model': model_id,
                    'Queries': model_stats['count'],
                    'Total Cost': model_stats['cost'],
                    'Avg Cost': model_stats['avg_cost'],
                    '% of Queries': (model_stats['count'] / stats['total_queries'] * 100) if stats['total_queries'] > 0 else 0,
                    '% of Cost': (model_stats['cost'] / stats['total_cost'] * 100) if stats['total_cost'] > 0 else 0
                })
            
            df = pd.DataFrame(comparison_data)
            
            # Display table
            st.dataframe(
                df.style.format({
                    'Total Cost': '${:.4f}',
                    'Avg Cost': '${:.4f}',
                    '% of Queries': '{:.1f}%',
                    '% of Cost': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Efficiency scatter plot
            st.subheader("Cost Efficiency Analysis")
            fig = px.scatter(
                df,
                x='% of Queries',
                y='% of Cost',
                size='Queries',
                color='Model',
                hover_data=['Avg Cost'],
                title='Model Efficiency (Lower is better)',
                labels={
                    '% of Queries': 'Percentage of Total Queries',
                    '% of Cost': 'Percentage of Total Cost'
                }
            )
            
            # Add ideal line (y=x)
            fig.add_shape(
                type='line',
                x0=0, y0=0, x1=100, y1=100,
                line=dict(color='gray', dash='dash'),
                name='Ideal (Cost = Usage)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:** 
            - Points **below** the diagonal line are cost-efficient (using less cost than expected)
            - Points **above** the diagonal are expensive (using more cost than their query share)
            """)
        else:
            st.info("No model performance data yet. Run some queries first!")