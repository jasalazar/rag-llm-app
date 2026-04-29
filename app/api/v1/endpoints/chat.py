from fastapi import APIRouter, HTTPException

from app.core.rag import query_rag
from app.schemas.models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    answer = await query_rag(request.question)
    return ChatResponse(answer=answer)
