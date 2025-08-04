# rag_pipeline.py
import requests
import pdfplumber
import faiss
import pickle
import numpy as np
import tempfile
from pathlib import Path
from openai import OpenAI
from utils import chunk_text, clean_text, table_to_markdown, save_chunks_to_json, truncate_chunks_to_token_limit
from responder import generate_structured_answer

# Load OpenAI key from .env (make sure it's loaded before import)
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("RAY_OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = "omniscient.index"
META_PATH = "metadata.pkl"


def download_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF from URL")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    return tmp.name

# def extract_text_from_pdf(pdf_path):
#     all_text = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text() or ""
#             if text.strip():
#                 all_text.append(f"[PAGE {i+1}]:\n{text.strip()}")
#     return "\n\n".join(all_text)

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text and tables from a PDF and returns as one big string.
    - Tables are appended as markdown tables for chunking/QA.
    """
    all_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. Extract normal text
            text = page.extract_text() or ""
            if text.strip():
                all_content.append(f"\n[PAGE {i+1} TEXT]\n{text.strip()}")

            # 2. Extract tables
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                # Convert table to markdown for readability & chunking
                if table:
                    table_md = table_to_markdown(table)
                    all_content.append(
                        f"\n[PAGE {i+1} TABLE {t_idx+1}]\n{table_md}"
                    )
    return "\n".join(all_content)


def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

# def build_faiss_index(chunks):
#     embeddings = [get_embedding(chunk['content']) for chunk in chunks]
#     dim = len(embeddings[0])
#     index = faiss.IndexHNSWFlat(dim, 32)
    
#     index.hnsw.efConstruction = 200
    
#     index.add(np.array(embeddings).astype("float32"))
    
#     with open(META_PATH, "wb") as f:
#         pickle.dump(chunks, f)
#     faiss.write_index(index, INDEX_PATH)
#     return index

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def build_faiss_index(chunks):
    safe_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk["content"])
        if len(tokens) <= 8192:
            safe_chunks.append(chunk)
        else:
            print(f"⚠️ Skipping overlong chunk: {chunk['chunk_id']} ({len(tokens)} tokens)")
    
    embeddings = [get_embedding(c["content"]) for c in safe_chunks]
    dim = len(embeddings[0])
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(np.array(embeddings).astype("float32"))

    with open(META_PATH, "wb") as f:
        pickle.dump(safe_chunks, f)
    faiss.write_index(index, INDEX_PATH)
    return index


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def rerank_chunks(chunks, query_embed):
    """
    Rerank retrieved chunks based on cosine similarity with query embedding.
    """
    def cosine_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    scored_chunks = []
    for chunk in chunks:
        chunk_embed = get_embedding(chunk["content"])
        score = cosine_sim(query_embed, chunk_embed)
        scored_chunks.append((chunk, score))

    # Sort descending and return top 5
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:5]]


# def search(query, top_k=5):
#     index = faiss.read_index("omniscient.index")
#     with open("metadata.pkl", "rb") as f:
#         metadata = pickle.load(f)
#     query_vec = get_embedding(query)
#     D, I = index.search(np.array([query_vec]).astype("float32"), top_k)
#     return [metadata[i] for i in I[0]]

def search(query, top_k=15):
    index = faiss.read_index(INDEX_PATH)
    index.hnsw.efSearch = 128
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    query_vec = get_embedding(query)
    D, I = index.search(np.array([query_vec]).astype("float32"), top_k)
    retrieved_chunks = [metadata[i] for i in I[0]]

    # 🔁 Apply semantic reranking
    return rerank_chunks(retrieved_chunks, query_vec)


def run_pipeline(url, questions):
    if not (Path("omniscient.index").exists()):
        pdf_path = download_pdf_from_url(url)
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        structured_chunks = [
            {"chunk_id": f"chunk_{i+1}_pg()", "content": chunk} for i, chunk in enumerate(chunks)
        ]
        
        save_chunks_to_json(structured_chunks) 
        index = build_faiss_index(structured_chunks)

    answers = []
    for q in questions:
        top_chunks = search(q)
        top_chunks = truncate_chunks_to_token_limit(top_chunks, max_total_tokens=6000)
        answer = generate_structured_answer(q, top_chunks)
        answers.append(answer)
    return answers
