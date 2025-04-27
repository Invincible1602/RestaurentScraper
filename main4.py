import os
import requests
import faiss
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

st.set_page_config(page_title="Restaurant RAG Chatbot", layout="centered")

dotenv_loaded = load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
INDEX_PATH = "csv_vector_index.faiss"
DATA_PATH = "csv_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RELEVANCE_THRESHOLD = 10.0
TOP_K = 5

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def load_faiss_index(index_path, data_path):
    if not (os.path.exists(index_path) and os.path.exists(data_path)):
        st.error("FAISS index or metadata file is missing.")
        return None, []
    index = faiss.read_index(index_path)
    with open(data_path, "rb") as f:
        text_data = pickle.load(f)
    if not isinstance(index, faiss.IndexFlatL2):
        st.error("Loaded FAISS index is not a valid IndexFlatL2 object.")
        return None, []
    return index, text_data

model = load_embedding_model(EMBEDDING_MODEL)
index, text_data = load_faiss_index(INDEX_PATH, DATA_PATH)

def search_similar_text(query):
    if index is None:
        return []
    query_embedding = model.encode([query], convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, TOP_K)
    return [text_data[i] for i, d in zip(indices[0], distances[0]) if d <= RELEVANCE_THRESHOLD]

def query_huggingface(prompt):
    if not HF_API_KEY:
        return "Error: Hugging Face API key missing."
    retrieved = search_similar_text(prompt)
    lower_q = prompt.lower()
    if any(term in lower_q for term in ["non vegetarian", "non-veg", "non veg"]):
        filtered = [text for text in retrieved if "vegetarian" not in text.lower()]
        if filtered:
            retrieved = filtered
    if not retrieved:
        return "I’m specialized in restaurant information, so I’m not able to recommend movies."
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
