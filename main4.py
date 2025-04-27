import os
# Disable TensorFlow in Hugging Face Transformers to prevent METAL plugin conflicts on Mac
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import streamlit as st
# Set page config first to avoid StreamlitSetPageConfigMustBeFirstCommandError
st.set_page_config(page_title="Restaurant RAG Chatbot", layout="centered")

import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

# Load environment variables
dotenv_loaded = load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

INDEX_PATH = "csv_vector_index.faiss"
DATA_PATH = "csv_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RELEVANCE_THRESHOLD = 10.0  # similarity threshold
TOP_K = 5

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load SentenceTransformer model
@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

model = load_embedding_model(EMBEDDING_MODEL)

# Load FAISS index & metadata
@st.cache_resource
def load_faiss_index(index_path, data_path):
    if not (os.path.exists(index_path) and os.path.exists(data_path)):
        st.error("FAISS index or metadata file is missing.")
        return None, []
    index = faiss.read_index(index_path)
    with open(data_path, "rb") as f:
        text_data = pickle.load(f)
    if not isinstance(index, faiss.IndexFlatL2):
        st.error("Loaded FAISS index is not a valid IndexFlatL2 object")
        return None, []
    return index, text_data

index, text_data = load_faiss_index(INDEX_PATH, DATA_PATH)


def search_similar_text(query):
    if index is None:
        return []
    # Compute embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    distances, indices = index.search(query_embedding, TOP_K)
    # Filter strictly by threshold
    relevant = [text_data[i] for i, d in zip(indices[0], distances[0]) if d <= RELEVANCE_THRESHOLD]
    return relevant


def query_huggingface(prompt):
    if not HF_API_KEY:
        return "Error: Hugging Face API key missing."
    # Retrieve context
    retrieved = search_similar_text(prompt)
    # Handle non-vegetarian requests by filtering out vegetarian options
    lower_q = prompt.lower()
    if any(term in lower_q for term in ["non vegetarian", "non-veg", "non veg"]):
        filtered = [text for text in retrieved if "vegetarian" not in text.lower()]
        if filtered:
            retrieved = filtered
    if not retrieved:
        # Out-of-scope fallback
        return "I’m specialized in restaurant information, so I’m not able to recommend movies."  
    # Format context
    context = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(retrieved)])
    full_prompt = (
        "You are an expert food and restaurant guide. "
        "Use the provided context to answer clearly and concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {prompt}\n"
        "Give a helpful structured response."
    )
    payload = {"inputs": full_prompt}
    try:
        res = requests.post(HF_API_URL, headers={"Authorization": f"Bearer {HF_API_KEY}"}, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        return "Failed to generate response."
    except Exception as e:
        logging.error(f"HF API error: {e}")
        return "Error querying language model."

# Streamlit UI
st.title("🍽️ Restaurant RAG Chatbot")
st.write("Ask questions about restaurant menus, prices, dietary options, and more.")

user_input = st.text_input("Your question:")
if st.button("Get Answer") and user_input:
    with st.spinner("Searching for answers..."):
        answer = query_huggingface(user_input)
    st.subheader("Response")
    st.write(answer)

st.write("---")
st.write("Built with Streamlit | Uses FAISS + Hugging Face RAG")
