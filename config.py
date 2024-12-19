from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
    QUERY_EMBED_MODEL_NAME = os.getenv("QUERY_EMBED_MODEL_NAME", "pritamdeka/S-BioBert-snli-multinli-stsb")
    CHATBOT_MODEL_NAME = os.getenv("CHATBOT_MODEL_NAME", "gpt-4")
