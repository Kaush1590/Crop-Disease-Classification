# Import required libraries

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit application

st.set_page_config(
    page_title="Potato Disease Dashboard",
    page_icon="🗃️",
    layout="wide"
)

st.title("Dashboard")

# Load model artifacts and evaluation data

status_path = Path(__file__).parent.parent / "model" / "model_stats.pkl"
variables_path = Path(__file__).parent.parent / "model" / "variables.pkl"

history = joblib.load(status_path)
variables = joblib.load(variables_path)

y_true = variables["y_true"]
y_pred = variables["y_pred"]

# Compute evaluation metrics

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "📈 Training",
        "🧩 Confusion Matrix",
        "⚙️ Dataset Info"
    ]
)

# Display model performance metrics

with tab1:
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    st.subheader("Prediction Distribution")

    pred_df = pd.DataFrame({
        "Predicted Class": y_pred
    })

    pred_counts = (
        pred_df["Predicted Class"]
        .value_counts()
        .reset_index()
    )

    pred_counts.columns = [
        "Class",
        "Count"
    ]

    fig_pred = px.bar(
        pred_counts,
        x="Class",
        y="Count",
        text="Count",
        title="Prediction Distribution"
    )

    st.plotly_chart(
        fig_pred,
        use_container_width=True
    )

# Analyze training performance

with tab2:
    st.subheader("📈 Training Performance")

    best_epoch = np.argmax(history["val_accuracy"]) + 1
    best_val_accuracy = max(history["val_accuracy"])
    lowest_val_loss = min(history["val_loss"])
    final_train_accuracy = history["accuracy"][-1]
    final_val_accuracy = history["val_accuracy"][-1]
    train_val_gap = abs(
        final_train_accuracy - final_val_accuracy
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    # Calculate training statistics

    metric_col1.metric(
        "Best Epoch",
        best_epoch
    )

    metric_col2.metric(
        "Best Validation Accuracy",
        f"{best_val_accuracy * 100:.2f}%"
    )

    metric_col3.metric(
        "Lowest Validation Loss",
        f"{lowest_val_loss:.4f}"
    )

    metric_col4.metric(
        "Train-Val Gap",
        f"{train_val_gap * 100:.2f}%"
    )

    st.divider()

    st.subheader("Model Information")

    # Display model configuration

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model",
            variables["model_name"]
        )

        st.metric(
            "Transfer Learning",
            "Enabled" if variables["transfer_learning"] else "Disabled"
        )

    with col2:
        st.metric(
            "Fine-Tuned Layers",
            variables["fine_tuned_layers"]
        )

        st.metric(
            "Learning Rate",
            variables["learning_rate"]
        )

    st.subheader("Model Generalization")

    # Assess model generalization

    if train_val_gap < 0.03:
        st.success("Model generalizes well with minimal overfitting.")

    elif train_val_gap < 0.08:
        st.warning("Minor overfitting detected.")

    else:
        st.error("Potential overfitting detected.")

    st.divider()

    st.subheader("Trainig curve")

    # Plot training and validation metrics

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_acc = go.Figure()
        fig_acc.add_trace(
            go.Scatter(
                x=np.arange(
                    1,
                    len(history["accuracy"]) + 1
                ),
                y=history["accuracy"],
                mode="lines+markers",
                name="Training Accuracy"
            )
        )

        fig_acc.add_trace(
            go.Scatter(
                x=np.arange(
                    1,
                    len(history["val_accuracy"]) + 1
                ),
                y=history["val_accuracy"],
                mode="lines+markers",
                name="Validation Accuracy"
            )
        )

        fig_acc.update_layout(
            title="Accuracy vs Epochs",
            xaxis_title="Epoch",
            yaxis_title="Accuracy"
        )

        st.plotly_chart(
            fig_acc,
            use_container_width=True
        )

    with chart_col2:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=np.arange(
                    1,
                    len(history["loss"]) + 1
                ),
                y=history["loss"],
                mode="lines+markers",
                name="Training Loss"
            )
        )

        fig_loss.add_trace(
            go.Scatter(
                x=np.arange(
                    1,
                    len(history["val_loss"]) + 1
                ),
                y=history["val_loss"],
                mode="lines+markers",
                name="Validation Loss"
            )
        )

        fig_loss.update_layout(
            title="Loss vs Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )

        st.plotly_chart(
            fig_loss,
            use_container_width=True
        )

    st.divider()

    st.subheader("Training Insights")

    # Summarize key training insights

    accuracy_gain = (
        history["val_accuracy"][-1]
        - history["val_accuracy"][0]
    )

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.info(
            f"""
            **Best Epoch:** {best_epoch}

            **Final Validation Accuracy:** {final_val_accuracy * 100:.2f}%

            **Validation Accuracy Gain:** {accuracy_gain * 100:.2f}%
            """
        )

    with insight_col2:
        st.info(
            f"""
            **Lowest Validation Loss:** {lowest_val_loss:.4f}

            **Final Training Accuracy:** {final_train_accuracy * 100:.2f}%

            **Train-Validation Gap:** {train_val_gap * 100:.2f}%
            """
        )

    st.divider()

    st.subheader("Epoch Statistics")

    # Display epoch metrics

    epoch_df = pd.DataFrame(
        {
            "Epoch":
                np.arange(
                    1,
                    len(history["accuracy"]) + 1
                ),

            "Train Accuracy":
                history["accuracy"],

            "Validation Accuracy":
                history["val_accuracy"],

            "Train Loss":
                history["loss"],

            "Validation Loss":
                history["val_loss"]
        }
    )

    st.dataframe(
        epoch_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

# Display classification results and confusion matrix

with tab3:
    report_df = pd.DataFrame(
        variables["classification_report"]
    ).transpose()

    st.subheader("Classification Report")

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
        )
    )

    fig_cm.update_layout(
        height=600
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

# Display dataset and project details

with tab4:

    st.subheader("Dataset Overview")

    st.markdown("""
        #### Project:
        [Potato Leaf Disease Dataset](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset)

        #### Model:
        MobileNetV2 Transfer Learning

        #### Framework:
        TensorFlow

        #### Deployment:
        Streamlit

        #### Classes:
        - Healthy
        - Early Blight
        - Late Blight
    """)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
            **Image Size:** {variables['image_size']} × {variables['image_size']}

            **Channels:** {variables['channel']}

            **Batch Size:** {variables['batch_size']}
            """
        )

    with col2:
        st.info(
            f"""
            **Epochs:** {variables['epochs']}

            **Training Samples:** {variables['training_len']}

            **Validation Samples:** {variables['validation_len']}

            **Testing Samples:** {variables['testing_len']}
            """
        )

    st.divider()

    st.subheader("Dataset Split")

    split_df = pd.DataFrame({
        "Dataset": [
            "Training",
            "Validation",
            "Testing"
        ],
        "Samples": [
            variables["training_len"],
            variables["validation_len"],
            variables["testing_len"]
        ]
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_pie = px.pie(
            split_df,
            names="Dataset",
            values="Samples",
            hole=0.4,
            title="Dataset Distribution"
        )

        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:
        total_samples = (
            variables["training_len"]
            + variables["validation_len"]
            + variables["testing_len"]
        )

        st.metric(
            "Total Samples",
            total_samples
        )

        st.metric(
            "Training %",
            f"{variables['training_len'] / total_samples * 100:.1f}%"
        )

        st.metric(
            "Validation %",
            f"{variables['validation_len'] / total_samples * 100:.1f}%"
        )

        st.metric(
            "Testing %",
            f"{variables['testing_len'] / total_samples * 100:.1f}%"
        )