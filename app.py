import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import altair as alt

st.set_page_config(page_title="Marathi Sentence Similarity", layout="wide")

# --- Title & Intro ---
st.title("Evaluating & Enhancing Marathi Sentence Similarity")
st.markdown("An interactive exploration of adapting AI for a low-resource language.")

# --- Load Fine‑Tuned Model ---
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("sangambhamare/MarathiSentenceSimilarity")

model = load_model()

# --- Phase 2: Fine-tuning Results (Static) ---
st.header("Performance Improvement After Fine-Tuning")
finetune_data = pd.DataFrame({
    "Metric": ["MSE", "MAE", "Pearson", "Spearman", "Accuracy (±0.1)", "Collision Rate"],
    "Baseline": [0.0232, 0.1181, 0.8722, 0.8549, 0.5249, 0.2712],
    "Fine-tuned": [0.0036, 0.0457, 0.9830, 0.9802, 0.9134, 0.2503]
})

chart_ft = (
    finetune_data.melt(id_vars="Metric", var_name="Version", value_name="Score")
    .pipe(lambda d: alt.Chart(d)
          .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
          .encode(
              x=alt.X("Metric:N", sort=None, title=None),
              y=alt.Y("Score:Q", title="Score"),
              column=alt.Column("Version:N", title=None,
                                header=alt.Header(labelAngle=0)),
          )
          .properties(height=400)
    )
)
st.altair_chart(chart_ft, use_container_width=True)

# --- Phase 3: Robustness Results (Static) ---
st.header("Robustness on Clean vs. Noisy Data")
robust_data = pd.DataFrame({
    "Dataset": ["Clean", "Basic Errors", "Advanced Errors"],
    "Baseline": [0.4673, 0.4623, 0.4121],
    "Fine-tuned": [0.5226, 0.5477, 0.4774]
})

chart_rb = (
    robust_data.melt(id_vars="Dataset", var_name="Version", value_name="Accuracy")
    .assign(Accuracy=lambda df: df.Accuracy * 100)
    .pipe(lambda d: alt.Chart(d)
          .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
          .encode(
              x=alt.X("Dataset:N", sort=None, title=None),
              y=alt.Y("Accuracy:Q", title="Accuracy (%)"),
              color=alt.Color("Version:N", title=None,
                              scale=alt.Scale(range=["#a0a0a0", "#7b9e89"]))
          )
          .properties(width=700, height=400)
    )
)
st.altair_chart(chart_rb, use_container_width=True)

# --- Interactive Similarity Calculator ---
st.header("Real-time Marathi Sentence Similarity Calculator")
s1 = st.text_area("First Marathi sentence", "", height=100)
s2 = st.text_area("Second Marathi sentence", "", height=100)

if st.button("Calculate Similarity"):
    if not s1.strip() or not s2.strip():
        st.warning("Please enter both Marathi sentences.")
    else:
        with st.spinner("Calculating…"):
            emb = model.encode([s1, s2])
            score = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))
        st.success(f"**Similarity Score:** {score:.2f}")

# --- Footer ---
st.markdown("---")
st.caption("Interactive report created by Sangam Sanjay Bhamare.")
