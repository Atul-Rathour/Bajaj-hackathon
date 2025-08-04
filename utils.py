import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def clean_text(raw_text: str) -> str:
    cleaned = raw_text
    # Company and policy meta (case-insensitive)
    cleaned = re.sub(r"Premises No.*\n?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"Registered Office.*\n?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"Policy Wordings.*\n?", "", cleaned, flags=re.I)
    # Contact info
    cleaned = re.sub(r"Tel:.*\n?", "", cleaned)
    cleaned = re.sub(r"Toll free:.*\n?", "", cleaned)
    cleaned = re.sub(r"Fax:.*\n?", "", cleaned)
    cleaned = re.sub(r"E-mail:.*\n?", "", cleaned)
    cleaned = re.sub(r"website:.*\n?", "", cleaned)
    # Reg numbers
    cleaned = re.sub(r"UIN:.*\n?", "", cleaned)
    cleaned = re.sub(r"CIN:.*\n?", "", cleaned)
    cleaned = re.sub(r"IRDAI Regn.*\n?", "", cleaned)
    cleaned = re.sub(r"PAN .*?\n?", "", cleaned)
    cleaned = re.sub(r"GSTIN.*?\n?", "", cleaned)
    # PDF meta-artifacts
    cleaned = re.sub(r"\[PAGE.*?\]", "", cleaned)
    cleaned = re.sub(r"\[IMAGE.*?\]", "", cleaned)
    # Page number refs
    cleaned = re.sub(r"Page\s*\d+\s*of\s*\d+", "", cleaned)
    cleaned = re.sub(r"Page\s*\d+", "", cleaned)
    # Remove ALL CAPS lines except section names
    cleaned = "\n".join([
        line for line in cleaned.splitlines()
        if not (line.strip().isupper() and
                line.strip() not in ["DEFINITIONS", "EXCLUSIONS", "BENEFITS", "COVERAGE"])
    ])
    # Collapse extra newlines/spaces
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def recursive_splitter(text: str, max_chars: int = 500, overlap: int = 150) -> list:
    """
    Smart semantic chunking using LangChain RecursiveCharacterTextSplitter.
    - Breaks text based on section headings, paragraphs, then sentences.
    - Prevents breaking mid-clause for legal documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = splitter.create_documents([text])
    return [doc.page_content.strip() for doc in documents]

from transformers import GPT2TokenizerFast
# Load tokenizer only once
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def token_based_chunking(text: str, max_tokens: int = 500, overlap: int = 100) -> list:
    """
    Token-aware chunking for better GPT alignment.
    Ensures no chunk exceeds max_tokens.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk.strip())
        start += max_tokens - overlap
    return chunks

from transformers import GPT2TokenizerFast
import re

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# def clause_aware_chunking(text, max_tokens=500, overlap=100):
#     clauses = re.split(r"(?=\n?\d+\.\d+(?:\.\d+)?[\.\s:–-])", text)
#     clauses = [c.strip() for c in clauses if c.strip()]

#     chunks = []
#     current_chunk = []
#     current_tokens = 0

#     for clause in clauses:
#         clause_tokens = tokenizer.encode(clause)
#         clause_len = len(clause_tokens)

#         if clause_len > max_tokens:
#             # ⚠️ Truncate oversize single clause
#             clause_tokens = clause_tokens[:max_tokens]
#             clause = tokenizer.decode(clause_tokens)

#         if current_tokens + clause_len <= max_tokens:
#             current_chunk.append(clause)
#             current_tokens += clause_len
#         else:
#             # ✅ Add the current chunk
#             chunks.append("\n".join(current_chunk))

#             # ✅ Overlap logic — keep last few tokens
#             overlap_tokens = []
#             overlap_count = 0
#             while current_chunk and overlap_count < overlap:
#                 ctok = tokenizer.encode(current_chunk[-1])
#                 overlap_tokens = ctok + overlap_tokens
#                 overlap_count += len(ctok)
#                 current_chunk.pop()

#             # Decode and start next chunk
#             current_chunk = [tokenizer.decode(overlap_tokens), clause]
#             current_tokens = len(tokenizer.encode(current_chunk[0] + clause))

#     if current_chunk:
#         chunks.append("\n".join(current_chunk))

#     return chunks

def clause_aware_chunking(text, max_tokens=500, overlap=100):
    """
    Splits the text into clause-aware chunks based on numbering patterns (e.g., 1.2.3)
    while ensuring that no chunk exceeds `max_tokens` and maintains `overlap` continuity.
    """
    # Smart regex to match common legal/insurance clauses like 1., 1.1, 2.3.5:
    clause_pattern = r"(?=\n?\s*\d+(\.\d+)*[\.\s:–-])"
    clauses = re.split(clause_pattern, text)
    clauses = [c.strip() for c in clauses if c.strip()]

    chunks = []
    current_chunk = []
    current_token_count = 0

    def encode(text):
        return tokenizer.encode(text)

    def decode(tokens):
        return tokenizer.decode(tokens)

    for clause in clauses:
        clause_tokens = encode(clause)
        clause_len = len(clause_tokens)

        if clause_len > max_tokens:
            # 🔥 Handle rare giant clause: split into subchunks safely
            subchunks = [
                clause_tokens[i:i+max_tokens]
                for i in range(0, clause_len, max_tokens)
            ]
            for sub in subchunks:
                sub_text = decode(sub)
                chunks.append(sub_text.strip())
            continue

        if current_token_count + clause_len <= max_tokens:
            current_chunk.append(clause)
            current_token_count += clause_len
        else:
            # ✅ Finalize current chunk
            chunks.append("\n".join(current_chunk).strip())

            # ➕ Overlap logic: re-encode last few clauses until overlap tokens met
            overlap_clauses = []
            overlap_token_count = 0

            for prev_clause in reversed(current_chunk):
                tok = encode(prev_clause)
                if overlap_token_count + len(tok) > overlap:
                    break
                overlap_clauses.insert(0, prev_clause)
                overlap_token_count += len(tok)

            # 🧱 Start new chunk with overlap + current clause
            current_chunk = overlap_clauses + [clause]
            current_token_count = sum(len(encode(c)) for c in current_chunk)

    # Append remaining chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return chunks



def chunk_text(text, strategy="clause", max_tokens=500, overlap=100):
    if strategy == "token":
        return token_based_chunking(text, max_tokens, overlap)
    elif strategy == "recursive":
        return recursive_splitter(text, max_chars=700, overlap=200)
    elif strategy == "clause":
        return clause_aware_chunking(text, max_tokens, overlap)


def table_to_markdown(table):
    """
    Converts pdfplumber list-of-lists table to markdown (optional).
    """
    if not table or not table[0]:
        return ""
    header = "| " + " | ".join(str(c).strip() for c in table[0]) + " |"
    sep = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows = ["| " + " | ".join(str(c).strip() for c in row) + " |" for row in table[1:]]
    return "\n".join([header, sep] + rows)

def save_chunks_to_json(data, filename="structured_chunks.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
        
def truncate_chunks_to_token_limit(chunks, max_total_tokens=6000):
    """
    Truncate total token count across all chunks to fit within GPT limit.
    """
    truncated = []
    total_tokens = 0

    for chunk in chunks:
        tokens = tokenizer.encode(chunk["content"])
        if total_tokens + len(tokens) > max_total_tokens:
            break
        truncated.append(chunk)
        total_tokens += len(tokens)

    return truncated
