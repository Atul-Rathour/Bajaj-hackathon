from openai import OpenAI
import os
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
    Formats a prompt for LLM with top-k chunks, returns only the answer as a string.
    """
    context_text = "\n---\n".join([f"[{c['chunk_id']}]\n{c['content']}" for c in chunks])

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
    - Avoid speculation or generic responses.
    - Use exact terms, clause language, or limits (like days, months, ₹ values) mentioned.
    - If the answer depends on conditions (e.g., eligibility period, age, continuous coverage), clearly mention **all** such conditions.
    - Keep the answer concise, factual, and easy to verify.
    - Format the answer to match the style of:
      - "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
      - "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    - Return ONLY the answer as a string, without reasoning, clauses, or the question.
    - Do NOT include quotation marks around the answer.

    Respond with the answer string directly.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        # Remove any surrounding quotes
        answer = answer.strip('"\'')
        return answer
    except Exception as e:
        return f"Error processing question: {str(e)}"