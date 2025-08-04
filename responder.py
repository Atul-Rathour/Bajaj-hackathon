# responder.py
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("RAY_OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o"


def generate_structured_answer(question, chunks):
    
    print("="*80)
    print(f"🔍 DEBUGGING FOR QUESTION:\n{question}\n")
    print("📄 Top Retrieved Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Chunk {i+1}: {chunk['chunk_id']}")
        print("-" * 60)
        print(chunk["content"])
        print("-" * 60)

    
    """
    Formats a prompt for LLM with top-k chunks, returns structured JSON with:
    - answer
    - reasoning
    - clause/chunk references
    """
    context_text = "\n---\n".join([f"[{c['chunk_id']}]\n{c['content']}" for c in chunks])

    # prompt = f"""
    #         You are an expert in analyzing legal, insurance, and compliance policy documents.

    #         Use the chunks provided below to answer the user question as accurately as possible.
    #         Do NOT use any external knowledge — rely ONLY on the given text.

    #         ---
    #         Question:
    #         {question}

    #         Relevant Clauses:
    #         {context_text}
    #         ---

    #        Instructions:
    #         - Extract the answer based only on the chunks.
    #         - Your answer MUST directly quote or summarize the clause in the chunks.
    #         - If clause mentions numbers, conditions, or waiting periods, include them exactly.
    #         - Use **precise legal terms** when available.
    #         - Add clause IDs like ["chunk_1_pg()", ...] that were most useful.
    #         - If no answer is found in the chunks, say: "Not found in provided clauses."

    #         Only trust the below clause content. Avoid vague summaries.

    #         ---

    #         Respond in the following JSON format:
    #         {{
    #             "question": "...",
    #             "answer": "...",           // concise and directly answers the question
    #             "reasoning": "...",        // explain why the chunks support the answer
    #             "clauses": ["chunk_3_pg_7", "chunk_6_pg_8"]
    #         }}
    #     """


    prompt = f"""
You are an expert in analyzing legal, insurance, and compliance policy documents.

Use the chunks provided below to answer the user question as accurately as possible.
Do NOT use any external knowledge — rely ONLY on the given text.

---
Question:
{question}

Relevant Clauses:
{context_text}
---

Instructions:
- Extract the answer based only on the chunks.
- Your answer MUST directly quote or summarize the clause in the chunks.
- If clause mentions numbers, conditions, or waiting periods, include them exactly.
- Use **precise legal terms** when available.
- Add clause IDs like ["chunk_1_pg()", ...] that were most useful.
- If no answer is found in the chunks, say: "Not found in provided clauses."

Only trust the below clause content. Avoid vague summaries.

---

Respond in the following JSON format:
{{
    "question": "{question}",
    "answer": "...",
    "reasoning": "...",
    "clauses": ["chunk_3_pg_7", "chunk_6_pg_8"]
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512
    )

    try:
        import json
        # text = response.choices[0].message.content.strip()
        # return json.loads(text)
        text = response.choices[0].message.content.strip()
        json_block = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_block)
    except Exception:
        return {
            "question": question,
            "answer": "Could not parse response.",
            "reasoning": "LLM returned unstructured text.",
            "clauses": [c["chunk_id"] for c in chunks]
        }
