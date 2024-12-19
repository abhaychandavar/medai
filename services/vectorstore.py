from config import Config
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.tools import create_retriever_tool

class VectorStore:
    _pinecone = None
    _index = None
    _vectorstore = None
    _retriever = None

    @staticmethod
    def init(api_key, env, index_name, embedding_model):
        """
        Initialize the VectorStore components. This method must be called once before using any other static methods.

        Args:
            api_key (str): The API key for Pinecone.
            env (str): The environment for Pinecone (e.g., "us-west1-gcp").
            index_name (str): The name of the Pinecone index.
            embedding_model (object): The embedding model to use for vectorization.
        """
        if VectorStore._vectorstore is None:  # Ensure initialization happens only once
            VectorStore._pinecone = Pinecone(api_key=api_key, environment=env)
            VectorStore._index = VectorStore._pinecone.Index(name=index_name)
            VectorStore._vectorstore = PineconeVectorStore(index=VectorStore._index, embedding=embedding_model)
            VectorStore._retriever = VectorStore._vectorstore.as_retriever(search_type='mmr', search_kwargs={
                'k': 3,
                'lambda_mult': 0.7
            })

    @staticmethod
    def get_top_k_simantic_similar_docs(query_embedding, k):
        """
        Fetches the top K most semantically similar documents based on the given query embedding.

        Args:
            query_embedding (list or np.array): The query embedding, typically a list or array of numbers representing the vector to search for similar documents.
            k (int): The number of top similar documents to return.

        Returns:
            list: A list of dictionaries, each containing:
                - "text" (str): The text of the document metadata.
                - "score" (float): The similarity score for the document, indicating how similar it is to the query embedding.
            
            Example return format:
            [
                {"text": "Document 1 text", "score": 0.95},
                {"text": "Document 2 text", "score": 0.92},
                {"text": "Document 3 text", "score": 0.89}
            ]
        """
        if VectorStore._retriever is None:
            raise RuntimeError("VectorStore is not initialized. Call initialize() first.")
        
        original_texts = []
        query_result = VectorStore._index.query(
                vector=[query_embedding],
                top_k=k,
                include_values=True
            )
        matches = query_result.get("matches", [])

        for match in matches:
            metadata_text = match.get("metadata", {}).get("text", "NA")
            score = match.get("score", 0.0)
            original_texts.append({"text": metadata_text, "score": score})

        return original_texts

    @staticmethod
    def get_vectorstore():
        """
        Get the current vector store instance.

        Returns:
            PineconeVectorStore: The vector store.
        """
        if VectorStore._vectorstore is None:
            raise RuntimeError("VectorStore is not initialized. Call initialize() first.")
        return VectorStore._vectorstore

    @staticmethod
    def get_retriever():
        """
        Get the current retriever instance.

        Returns:
            Retriever: The retriever used for semantic search.
        """
        if VectorStore._retriever is None:
            raise RuntimeError("VectorStore is not initialized. Call initialize() first.")
        return VectorStore._retriever
    
    @staticmethod
    def get_tool(name, description):
        retriever_tool = create_retriever_tool(retriever = VectorStore._retriever, 
                                       name = name,
                                      description = description)
        return retriever_tool