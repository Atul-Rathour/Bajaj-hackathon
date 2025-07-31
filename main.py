# main.py
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from rag_pipeline import run_pipeline
from dotenv import load_dotenv
import os
import uvicorn

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("HACKRX_API_KEY")

# Check that the API key is set
if not API_KEY:
    raise ValueError("HACKRX_API_KEY environment variable not set. Please set it.")

app = FastAPI()

class RAGRequest(BaseModel):
    documents: str  # PDF Blob URL
    questions: List[str]

@app.post("/hackrx/run")
async def run_rag(
    request: RAGRequest,
    authorization: Optional[str] = Header(None) # Making the header optional
):
    # Check if the Authorization header is missing
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing"
        )
    
    # Check if the header starts with "Bearer " and the token is valid
    # The f-string is a great way to handle this
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token"
        )

    # Run pipeline
    result = run_pipeline(request.documents, request.questions)
    return {"answers": result}

if __name__ == "__main__":
    # Your uvicorn command seems fine.
    # For production, you would remove reload=True
    uvicorn.run("main:app", host="0.0.0.0", port=8000)