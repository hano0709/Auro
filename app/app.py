from fastapi import FastAPI
from app.chatbot import base_chat, chatting, main
from app.models import *
import asyncio
import threading

app = FastAPI()

@app.get("/")
async def home_page():
    return {"message" : "hello hi world"}

@app.post("/base_chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = base_chat(req.message)
    return ChatResponse(response=reply)

@app.post("/chat_with_history")
async def chat_with_history(req: ChatWithHistoryRequest):
    generator = chatting(req.message, history=req.history)
    response = ""
    for chunk in generator:
        response = chunk
    return {"response": response}

@app.get("/gradio")
def gradio_ui():
    # Run Gradio in a separate thread to avoid blocking the FastAPI app
    threading.Thread(target=main).start()
    return {"message": "Gradio UI launched. Visit the link in terminal to access it."}