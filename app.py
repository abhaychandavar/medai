from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from services.vectorstore import VectorStore
from config import Config
from services.embed import Embed
from services.chatbot import Chatbot
from services.wikipedia import Wikipedia

def main():
    Wikipedia.init()
    Embed.init(Config.QUERY_EMBED_MODEL_NAME)
    embedding_model = Embed.get_embedding_model()
    VectorStore.init(Config.PINECONE_API_KEY, Config.PINECONE_ENV, Config.PINECONE_INDEX_NAME, embedding_model)
    Chatbot.init()
    agent = Chatbot.get_chatbot_agent()
    return agent

main()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

class QueryRequest(BaseModel):
    query: str

@app.post("/converse")
async def converse(request: QueryRequest):
    try:
        return StreamingResponse(Chatbot.invoke_agent(request.query), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print('Running medAI')
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT, reload=True)
