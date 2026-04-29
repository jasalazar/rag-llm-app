from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    text: str
    source: str = "uploaded"


class IngestResponse(BaseModel):
    doc_id: str
    message: str
