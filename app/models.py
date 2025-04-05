from pydantic import BaseModel 

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    
class ChatWithHistoryRequest(BaseModel):
    message: str
    history: List[str]