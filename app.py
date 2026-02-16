import streamlit as st

st.set_page_config(
    page_title="MLOps AutoML Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MLOps AutoML Platform")
st.markdown(
    """
    An end-to-end machine learning platform — upload data, preprocess, train models,
    compare results, serve predictions, and monitor for drift.

    **Get started** by navigating to a page in the sidebar.

    ---

    ### Workflow

    1. **Upload & Explore** — Load your dataset and review an auto-generated EDA report
    2. **Preprocessing** — Configure or auto-apply data transformations
    3. **Training** — Train Random Forest, XGBoost, or Neural Network models
    4. **Results** — Compare metrics, visualizations, and promote the best model
    5. **Prediction** — Run batch or single predictions with the registered model
    6. **Monitoring** — Track model health and detect data drift
    """
)
