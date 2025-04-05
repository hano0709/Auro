from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def home_page():
    return {"message" : "hello hi world"}