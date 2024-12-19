
from langchain_huggingface import HuggingFaceEmbeddings

class Embed:
    _model_name = None
    _embeddings = None

    @staticmethod
    def init(model_name):
        """
        Initialize the embeddings with the provided model name. This method must be called 
        once before using any other static methods.

        Args:
            model_name (str): The name of the model to use for HuggingFace embeddings.
        """
        if Embed._embeddings is None:
            Embed._model_name = model_name
            Embed._embeddings = HuggingFaceEmbeddings(model_name=model_name)

    @staticmethod
    def embed_text(text):
        """
        Embed the provided text using the pre-initialized embeddings model.

        Args:
            text (str): The text to be embedded.

        Returns:
            list: The embedded representation of the text.
        """
        if Embed._embeddings is None:
            raise RuntimeError("Embed is not initialized. Call initialize() first.")
        return Embed._embeddings.embed_query([text])

    @staticmethod
    def get_embedding_model():
        """
        Get the current embeddings model.

        Returns:
            HuggingFaceEmbeddings: The embeddings model.
        """
        if Embed._embeddings is None:
            raise RuntimeError("Embed is not initialized. Call initialize() first.")
        return Embed._embeddings