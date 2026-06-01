
import streamlit as st
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import plotly.express as px
import plotly.graph_objects as go

# ====================================================
# PAGE CONFIG
# ====================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ====================================================
# LOAD MODELS
# ====================================================

MAX_LEN = 200

@st.cache_resource
def load_all_models():
    rnn_model = tf.keras.models.load_model("simple_rnn.keras")
    lstm_model = tf.keras.models.load_model("lstm_model.keras")
    gru_model = tf.keras.models.load_model("gru_model.keras")

    return rnn_model, lstm_model, gru_model


rnn_model, lstm_model, gru_model = load_all_models()

# ====================================================
# LOAD TOKENIZER
# ====================================================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ====================================================
# PREDICTION FUNCTION
# ====================================================

def predict_review(review, model):

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    positive_prob = float(prediction)
    negative_prob = 1 - positive_prob

    sentiment = (
        "Positive"
        if positive_prob >= 0.5
        else "Negative"
    )

    confidence = max(
        positive_prob,
        negative_prob
    )

    return (
        sentiment,
        confidence,
        positive_prob,
        negative_prob
    )

# ====================================================
# HEADER
# ====================================================

st.markdown(
    """
    <h1 style='text-align:center;'>
    🎬 Movie Review Sentiment Analysis System
    </h1>

    <h4 style='text-align:center;color:gray;'>
    Deep Learning Based Sentiment Classification
    </h4>

    <hr>
    """,
    unsafe_allow_html=True
)

# ====================================================
# SIDEBAR
# ====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Single Model Prediction",
        "Model Comparison"
    ]
)

# ====================================================
# PAGE 1
# ====================================================

if page == "Single Model Prediction":

    st.subheader("Analyze a Movie Review")

    selected_model = st.selectbox(
        "Choose Model",
        [
            "SimpleRNN",
            "LSTM",
            "GRU"
        ]
    )

    review = st.text_area(
        "Enter your movie review here...",
        height=200
    )

    if st.button("🔍 Analyze Review"):

        if review.strip() == "":
            st.warning(
                "Please enter a review."
            )

        else:

            if selected_model == "SimpleRNN":
                model = rnn_model

            elif selected_model == "LSTM":
                model = lstm_model

            else:
                model = gru_model

            sentiment, confidence, pos, neg = \
                predict_review(review, model)

            # ===================================
            # SENTIMENT CARD
            # ===================================

            if sentiment == "Positive":

                st.markdown(
                    """
                    <div style="
                    background-color:#d4edda;
                    padding:20px;
                    border-radius:15px;
                    text-align:center;">
                    <h2>😊 Positive Review</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div style="
                    background-color:#f8d7da;
                    padding:20px;
                    border-radius:15px;
                    text-align:center;">
                    <h2>☹️ Negative Review</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # ===================================
            # METRICS
            # ===================================

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Sentiment",
                    sentiment
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

            # ===================================
            # PROGRESS BAR
            # ===================================

            st.write("### Confidence Score")

            st.progress(float(confidence))

            st.write(
                f"**Confidence Percentage:** "
                f"{confidence*100:.2f}%"
            )

            # ===================================
            # CHARTS
            # ===================================

            left, right = st.columns(2)

            with left:

                prob_df = pd.DataFrame({
                    "Sentiment": [
                        "Positive",
                        "Negative"
                    ],
                    "Probability": [
                        pos,
                        neg
                    ]
                })

                fig = px.pie(
                    prob_df,
                    values="Probability",
                    names="Sentiment",
                    hole=0.6,
                    title="Sentiment Probability Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with right:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence*100,
                        title={
                            "text":
                            "Confidence Meter"
                        },
                        gauge={
                            "axis": {
                                "range":[0,100]
                            }
                        }
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

# ====================================================
# PAGE 2
# ====================================================

elif page == "Model Comparison":

    st.subheader(
        "Compare All Models"
    )

    review = st.text_area(
        "Enter Review",
        height=200
    )

    if st.button("⚖ Compare Models"):

        if review.strip() == "":
            st.warning(
                "Please enter a review."
            )

        else:

            models = {
                "SimpleRNN": rnn_model,
                "LSTM": lstm_model,
                "GRU": gru_model
            }

            results = []

            for model_name, model in models.items():

                sentiment, confidence, pos, neg = \
                    predict_review(
                        review,
                        model
                    )

                results.append([
                    model_name,
                    sentiment,
                    round(
                        confidence*100,
                        2
                    )
                ])

            df = pd.DataFrame(
                results,
                columns=[
                    "Model",
                    "Prediction",
                    "Confidence (%)"
                ]
            )

            st.write(
                "### Comparison Results"
            )

            st.dataframe(
                df.style.highlight_max(
                    subset=[
                        "Confidence (%)"
                    ],
                    color="lightgreen"
                )
            )

            fig = px.bar(
                df,
                x="Model",
                y="Confidence (%)",
                color="Prediction",
                title="Confidence Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# ====================================================
# FOOTER
# ====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    Developed using TensorFlow • Keras • Streamlit • Deep Learning
    </center>
    """,
    unsafe_allow_html=True
)
