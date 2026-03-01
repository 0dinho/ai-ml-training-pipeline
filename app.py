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

    Supports **6 task types**: Binary Classification, Multiclass Classification,
    Regression, Clustering, Anomaly Detection, and Dimensionality Reduction.

    **Get started** by navigating to a page in the sidebar.

    ---

    ### Workflow

    1. **Upload & Explore** — Load your dataset and review an auto-generated EDA report
    2. **Preprocessing** — Configure or auto-apply data transformations and splitting
    3. **Feature Engineering** *(optional)* — Polynomial features, transforms, and binning
    4. **Training** — Train 20+ models across all supported task types
    5. **Results** — Compare metrics, visualizations, and promote the best model
    6. **Prediction** — Run batch or single predictions (cluster labels, anomaly scores, coordinates, or class probabilities)
    7. **Monitoring** — Track model health, detect data drift, and trigger retraining
    """
)
