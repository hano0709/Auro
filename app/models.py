from pydantic import BaseModel
from typing import List , Optional

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model: Optional[str] = None
class ChatResponse(BaseModel):
    response: str
    
class ChatWithHistoryRequest(BaseModel):
    message: str
    history: List[str]