from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import base_chat, chatting, main
from app.models import *
import gradio as gr
import asyncio
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

gradio_app = main()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")