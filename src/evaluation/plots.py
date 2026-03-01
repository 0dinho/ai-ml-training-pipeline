"""Pure Plotly visualization helpers for all 6 task types.

No Streamlit imports — all functions return go.Figure objects.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORWAY = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=COLORWAY,
)


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------


def cluster_scatter_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Scatter (2D)",
) -> go.Figure:
    """2D scatter plot colored by cluster label."""
    unique_labels = np.unique(labels)
    fig = go.Figure()
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        color = "#888888" if label == -1 else COLORWAY[i % len(COLORWAY)]
        fig.add_trace(
            go.Scatter(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode="markers",
                name=name,
                marker=dict(color=color, size=6, opacity=0.75),
            )
        )
    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2", **DARK_LAYOUT)
    return fig


def elbow_curve(k_values: list[int], inertias: list[float]) -> go.Figure:
    """KMeans elbow curve: inertia vs. k."""
    fig = go.Figure(
        go.Scatter(
            x=k_values,
            y=inertias,
            mode="lines+markers",
            marker=dict(color=COLORWAY[0], size=8),
            line=dict(color=COLORWAY[0]),
        )
    )
    fig.update_layout(
        title="Elbow Curve (KMeans)",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        **DARK_LAYOUT,
    )
    return fig


def silhouette_curve(k_values: list[int], scores: list[float]) -> go.Figure:
    """Silhouette score vs. k."""
    fig = go.Figure(
        go.Scatter(
            x=k_values,
            y=scores,
            mode="lines+markers",
            marker=dict(color=COLORWAY[1], size=8),
            line=dict(color=COLORWAY[1]),
        )
    )
    fig.update_layout(
        title="Silhouette Score vs. k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        **DARK_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly detection plots
# ---------------------------------------------------------------------------


def anomaly_score_histogram(
    scores: np.ndarray,
    threshold: float | None = None,
    title: str = "Anomaly Score Distribution",
) -> go.Figure:
    """Histogram of anomaly scores with an optional threshold line."""
    fig = go.Figure(
        go.Histogram(
            x=scores,
            nbinsx=40,
            marker_color=COLORWAY[0],
            opacity=0.75,
            name="Anomaly Scores",
        )
    )
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#FFEAA7",
            annotation_text=f"Threshold ({threshold:.3f})",
            annotation_position="top right",
        )
    fig.update_layout(title=title, xaxis_title="Score", yaxis_title="Count", **DARK_LAYOUT)
    return fig


def anomaly_scatter_2d(
    X_2d: np.ndarray,
    is_anomaly: np.ndarray,
    title: str = "Anomaly Detection (2D Projection)",
) -> go.Figure:
    """2D scatter highlighting anomalous points."""
    normal_mask = is_anomaly == 0
    anomaly_mask = is_anomaly == 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X_2d[normal_mask, 0],
            y=X_2d[normal_mask, 1],
            mode="markers",
            name="Normal",
            marker=dict(color=COLORWAY[1], size=5, opacity=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_2d[anomaly_mask, 0],
            y=X_2d[anomaly_mask, 1],
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLORWAY[0], size=8, symbol="x", opacity=0.9),
        )
    )
    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2", **DARK_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Dimensionality reduction plots
# ---------------------------------------------------------------------------


def reduction_scatter_2d(
    coords: np.ndarray,
    color_values: np.ndarray | None = None,
    color_label: str = "Target",
    title: str = "Embedding (2D)",
) -> go.Figure:
    """2D scatter of embedding coordinates, optionally colored by target."""
    fig = go.Figure(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                color=color_values if color_values is not None else COLORWAY[0],
                colorscale="Viridis" if color_values is not None else None,
                colorbar=dict(title=color_label) if color_values is not None else None,
                size=5,
                opacity=0.75,
            ),
        )
    )
    fig.update_layout(title=title, xaxis_title="Dim 1", yaxis_title="Dim 2", **DARK_LAYOUT)
    return fig


def reduction_scatter_3d(
    coords: np.ndarray,
    color_values: np.ndarray | None = None,
    color_label: str = "Target",
    title: str = "Embedding (3D)",
) -> go.Figure:
    """3D scatter of embedding coordinates."""
    if coords.shape[1] < 3:
        raise ValueError("Need at least 3 components for 3D scatter.")
    fig = go.Figure(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                color=color_values if color_values is not None else COLORWAY[0],
                colorscale="Viridis" if color_values is not None else None,
                colorbar=dict(title=color_label) if color_values is not None else None,
                size=3,
                opacity=0.7,
            ),
        )
    )
    fig.update_layout(title=title, **DARK_LAYOUT)
    return fig


def scree_plot(explained_variance_ratio: np.ndarray, title: str = "Scree Plot") -> go.Figure:
    """Bar chart of explained variance per component."""
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=list(range(1, n + 1)),
            y=explained_variance_ratio.tolist(),
            name="Per Component",
            marker_color=COLORWAY[0],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, n + 1)),
            y=cumulative.tolist(),
            mode="lines+markers",
            name="Cumulative",
            marker=dict(color=COLORWAY[1], size=6),
            line=dict(color=COLORWAY[1]),
        ),
        secondary_y=True,
    )
    fig.update_layout(title=title, xaxis_title="Principal Component", **DARK_LAYOUT)
    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Supervised: additional plots
# ---------------------------------------------------------------------------


def precision_recall_curve_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Precision-Recall Curve",
) -> go.Figure:
    """Precision-Recall curve for binary or multiclass (one-vs-rest)."""
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import label_binarize

    fig = go.Figure()

    unique = np.unique(y_true)
    if len(unique) == 2:
        # Binary
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name="PR Curve",
                line=dict(color=COLORWAY[0]),
            )
        )
    else:
        # Multiclass OvR
        y_bin = label_binarize(y_true, classes=unique)
        for i, cls in enumerate(unique):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            name = str(class_names[i]) if class_names else f"Class {cls}"
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=name,
                    line=dict(color=COLORWAY[i % len(COLORWAY)]),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    return fig
