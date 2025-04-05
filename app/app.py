from fastapi import FastAPI
from chatbot import base_chat
from models import *

app = FastAPI()

@app.get("/")
async def home_page():
    return {"message" : "hello hi world"}

@app.post("/base_chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = base_chat(req.message)
    return ChatResponse(response=reply)