from fastapi import FastAPI, WebSocket
from gpt4all import GPT4All

app = FastAPI()

model = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")

@app.websocket("/generate")
async def generate(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        for output in model.generate(data, max_tokens=2048, streaming=True):
            await websocket.send_text(output)
