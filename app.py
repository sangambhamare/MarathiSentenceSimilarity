import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import altair as alt

# Page configuration
st.set_page_config(page_title="Marathi Sentence Similarity", layout="wide")

# Title and description
st.title("Evaluating & Enhancing Marathi Sentence Similarity")
st.markdown(
    "An interactive exploration of adapting AI for a low-resource language."
)

# Load the fine-tuned model
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("sangambhamare/MarathiSentenceSimilarity")

model = load_model()

# Phase 2: Fine-tuning results (static example)
st.header("Performance After Fine-Tuning")
finetune_data = pd.DataFrame({
    "Metric": ["MSE", "MAE", "Pearson", "Spearman", "Accuracy (±0.1)", "Collision Rate"],
    "Baseline": [0.0232, 0.1181, 0.8722, 0.8549, 0.5249, 0.2712],
    "Fine-tuned": [0.0036, 0.0457, 0.9830, 0.9802, 0.9134, 0.2503]
})

melted = finetune_data.melt(id_vars="Metric", var_name="Model", value_name="Value")
# Convert percentages
melted["Display"] = np.where(
    melted["Metric"].str.contains("Accuracy|Collision"),
    melted["Value"] * 100,
    melted["Value"]
)

chart = (
    alt.Chart(melted)
    .mark_bar()
    .encode(
        x=alt.X("Metric:N", sort=None, title=None),
        y=alt.Y("Display:Q", title="Score"),
        color=alt.Color("Model:N", title=None,
                        scale=alt.Scale(range=["#a0a0a0", "#7b9e89"]))
    )
    .properties(width=700, height=400)
)

st.altair_chart(chart, use_container_width=True)

# Phase 3: Robustness results
st.header("Robustness on Noisy Data")
robust_df = pd.DataFrame({
    "Dataset": ["Clean Data", "Basic Errors", "Advanced Errors"],
    "Baseline": [0.4673, 0.4623, 0.4121],
    "Fine-tuned": [0.5226, 0.5477, 0.4774]
})

robust_melt = robust_df.melt(id_vars="Dataset", var_name="Model", value_name="Accuracy")
robust_melt["AccuracyPct"] = robust_melt["Accuracy"] * 100

chart2 = (
    alt.Chart(robust_melt)
    .mark_bar()
    .encode(
        x=alt.X("Dataset:N", sort=None, title=None),
        y=alt.Y("AccuracyPct:Q", title="Accuracy (%)"),
        color=alt.Color("Model:N", title=None,
                        scale=alt.Scale(range=["#a0a0a0", "#7b9e89"]))
    )
    .properties(width=700, height=400)
)

st.altair_chart(chart2, use_container_width=True)

# Interactive similarity calculator
st.header("Real-time Marathi Sentence Similarity Calculator")
sentence1 = st.text_area("Sentence 1", height=100)
sentence2 = st.text_area("Sentence 2", height=100)

if st.button("Calculate Similarity"):
    if not sentence1 or not sentence2:
        st.warning("Please enter both Marathi sentences.")
    else:
        with st.spinner("Calculating…"):
            emb = model.encode([sentence1, sentence2])
            score = float(
                np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
            )
        st.success(f"**Similarity Score:** {score:.2f}")

# Footer
st.markdown("---")
st.caption("Interactive report created by Sangam Sanjay Bhamare.")
