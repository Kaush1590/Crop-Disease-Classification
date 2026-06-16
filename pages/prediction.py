# Import required libraries

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

# Configure Streamlit application

st.set_page_config(
    page_title="Prediction",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Potato Leaf Disease Prediction")

# Load trained model

@st.cache_resource
def load_model():
    model_path = (
        Path(__file__).parent.parent / "model" / "model.keras"
    )

    return tf.keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input}
    )

model = load_model()

# Define model configuration

IMAGE_SIZE = 256
class_names = [
    "Potato___Early_Blight",
    "Potato___Late_Blight",
    "Potato___Healthy"
]

# Disease descriptions and recommendations

DISEASE_INFO = {
    "Potato___Early_Blight": {
        "Symptoms":
            "Dark brown circular spots with concentric rings on leaves.",

        "Treatment":
            "Apply fungicides and remove infected leaves.",

        "Prevention":
            "Use crop rotation and avoid overhead irrigation."
    },

    "Potato___Late_Blight": {
        "Symptoms":
            "Water-soaked lesions that rapidly spread and darken.",

        "Treatment":
            "Apply recommended fungicides immediately.",

        "Prevention":
            "Monitor humidity and improve field ventilation."
    },

    "Potato___Healthy": {
        "Symptoms":
            "No disease symptoms detected.",

        "Treatment":
            "No treatment required.",

        "Prevention":
            "Maintain regular crop management practices."
    }
}

# Upload image for prediction

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

# Preprocess uploaded image

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

    image_resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Generate prediction

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        np.max(predictions[0]) * 100
    )

    # Display prediction results

    with col2:
        st.subheader("Prediction Result")

        st.success(
            predicted_class.replace(
                "Potato___",
                ""
            )
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        if confidence >= 95:
            st.success(
                "High Confidence Prediction"
            )

        elif confidence >= 80:
            st.warning(
                "Moderate Confidence Prediction"
            )

        else:
            st.error(
                "Low Confidence Prediction"
            )

    st.divider()

    st.subheader(
        "Prediction Probabilities"
    )

    # Prepare prediction probabilities

    probability_df = pd.DataFrame({
        "Class": [
            name.replace(
                "Potato___",
                ""
            )
            for name in class_names
        ],
        "Probability":
            predictions[0] * 100
    })

    # Visualize class probabilities

    fig = px.bar(
        probability_df,
        x="Class",
        y="Probability",
        text="Probability",
        title="Class Probabilities"
    )

    fig.update_traces(
        texttemplate="%{y:.2f}%"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.divider()

    st.subheader(
        "Disease Information"
    )

    # Display disease information

    info = DISEASE_INFO[
        predicted_class
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            f"""
            **Symptoms**

            {info['Symptoms']}
            """
        )

    with col2:
        st.warning(
            f"""
            **Treatment**

            {info['Treatment']}
            """
        )

    with col3:
        st.success(
            f"""
            **Prevention**

            {info['Prevention']}
            """
        )

    st.divider()

    st.subheader(
        "Prediction Details"
    )

    # Show detailed prediction scores

    st.dataframe(
        probability_df.sort_values(
            by="Probability",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )