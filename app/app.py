from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import base_chat, chatting, main
from app.models import *
from dotenv import load_dotenv
import gradio as gr
import asyncio
import threading
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-pro"

@app.get("/")
async def home_page():
    return {"message" : "hello hi world"}

@app.post("/base_chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = base_chat(
        req.message,
        api_key=req.api_key or GEMINI_API_KEY,
        model=req.model or GEMINI_MODEL
    )
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