# Install dependencies before running
# !pip install streamlit sentence-transformers PyMuPDF faiss-cpu requests

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import pickle
import hashlib

# --- Groq API Setup ---
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = ""  # Replace this with your actual Groq API key
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# --- Cache Directory Setup ---
CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_path):
    """Generate a hash for the file to use as cache identifier"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_from_cache(file_hash):
    """Load embeddings and chunks from cache if they exist"""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(file_hash, data):
    """Save embeddings and chunks to cache"""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

# --- Helper Functions ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdf(file_path):
    """Process PDF and return chunks, embeddings, and index"""
    file_hash = get_file_hash(file_path)
    cached_data = load_from_cache(file_hash)
    
    if cached_data:
        st.info(" Using cached embeddings for this PDF")
        return cached_data['chunks'], cached_data['embeddings'], cached_data['index']
    
    with st.spinner(" Extracting text from PDF..."):
        text = extract_text_from_pdf(file_path)

    with st.spinner(" Splitting and embedding text..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text)

        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        chunk_embeddings = embed_model.encode(chunks)

        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(np.array(chunk_embeddings))
        
        # Save to cache
        save_to_cache(file_hash, {
            'chunks': chunks,
            'embeddings': chunk_embeddings,
            'index': index
        })
        
    return chunks, chunk_embeddings, index

def get_answer(query, chunks, embed_model, index):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    matched_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(matched_chunks)

    prompt = f"""Use the following context to answer the question in a detailed manner.

Context:
{context}

Question: {query}
Answer (Provide a detailed response based on the context above):"""

    data = {
        "model": "mistral-saba-24b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who gives accurate and detailed answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f" Error: {response.status_code} - {response.text}"

# --- Streamlit App ---
st.set_page_config(page_title="RAG-Based PDF Question-Answering System", layout="centered")
st.title("PDF Q&A with Mistral-SABA-24B (Groq)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Process PDF (will use cache if available)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks, chunk_embeddings, index = process_pdf(file_path)
    
    st.success(" PDF processed successfully! You can now ask questions.")

    query = st.text_input("Ask a question about the PDF content:")
    if query:
        with st.spinner(" Generating answer..."):
            answer = get_answer(query, chunks, embed_model, index)
        st.markdown(" Answer:")
        st.write(answer)

    # Clean up
    os.remove(file_path) 