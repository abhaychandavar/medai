import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.constants import Constants
from . import vectorstore 
from config import Config
from . import wikipedia
from langgraph.prebuilt import create_react_agent

class Chatbot:
    chat = None
    prompt_template = None
    vector_store = None
    chain = None
    search = None
    tools = None
    @staticmethod
    def init(model_name=Config.CHATBOT_MODEL_NAME, temperature=0, max_tokens=512, seed=42):
        if not Chatbot.chat:
            Chatbot.chat = ChatOpenAI(
                model_name=model_name,
                seed=seed,
                temperature=temperature,
                max_tokens=max_tokens
            )

            Chatbot.prompt_template = PromptTemplate.from_template(Constants.Template)

            Chatbot.vector_store = vectorstore.VectorStore.get_retriever()

            Chatbot.chain = {
                "context": Chatbot.vector_store,
                "question": RunnablePassthrough(),
            } | Chatbot.prompt_template | Chatbot.chat | StrOutputParser()

    @staticmethod
    def answer_question(question: str) -> str:
        """
        Answer a given question using the defined chain of components. This method requires
        that the class has been initialized first.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The response generated by the chain.
        """
        if not Chatbot.chain:
            raise RuntimeError("Chatbot is not initialized. Call initialize() first.")
        
        response = Chatbot.chain.invoke(question)
        return response

    @staticmethod
    def get_chatbot_agent ():
        wiki_tool = wikipedia.Wikipedia.get_tool()
        vec_tool = vectorstore.VectorStore.get_tool(name='medical-query-context-fetcher', description=''' 
                                      For any questions regarding medical queries such you must use this tool, and look for the answer in the information provided by this tool.
                                      ''')
        tools = [wiki_tool, vec_tool]
        agent = create_react_agent(
                    model = Chatbot.chat,
                    tools = tools,
                    messages_modifier = ''' You are Medira, a chatbot that answers medical questions politely, you can also answer generic questions. your gender is female
                    If you don't know the answer just say you don't know the answer and you are still learning
                    '''
                )
        res = agent.invoke({
            'messages': [
                {
                'role': "user",
                'content': 'Why do I have fever?',
                },
            ],
            })
        print(res['messages'][len(res['messages']) - 1])
        return agent
        
        
        

