from pydantic import BaseModel
from typing import List 

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    
class ChatWithHistoryRequest(BaseModel):
    message: str
    history: List[str]