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

chatbot = main()