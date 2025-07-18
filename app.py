# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import hf_hub_download

# Load the specified Marathi similarity model once at startup
@st.cache_resource
def load_model():
    try:
        # Use trust_remote_code if model repo contains custom code
        model = SentenceTransformer(
            'sangambhamare/MarathiSentenceSimilarity',
            trust_remote_code=True
        )
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()
    return model

model = load_model()

st.title("Marathi Sentence Similarity Calculator")

# Two text inputs for Marathi sentences
sent1 = st.text_area("पहिली मराठी वाक्यं टाका:", height=100)
sent2 = st.text_area("दुसरी मराठी वाक्यं टाका:", height=100)

# Button to trigger similarity computation
if st.button("समानता तपासा"):
    if not sent1.strip() or not sent2.strip():
        st.warning("कृपया दोन्ही वाक्यं भरा.")
    else:
        with st.spinner("समानता मोजत आहे..."):
            try:
                emb1 = model.encode(sent1, convert_to_tensor=True)
                emb2 = model.encode(sent2, convert_to_tensor=True)
                score = util.cos_sim(emb1, emb2).item()
            except Exception as e:
                st.error(f"Similarity computation failed: {e}")
                st.stop()

        # Display results
        st.markdown(f"**समानता गुणांक:** {score:.4f}")
        st.progress(min(max(score, 0.0), 1.0))

# Sidebar instructions
st.sidebar.header("सहाय्य")
