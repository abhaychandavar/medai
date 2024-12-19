from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

class Wikipedia:
    __wikipedia_api = None

    @staticmethod
    def init ():
        Wikipedia.__wikipedia_api = WikipediaAPIWrapper()
        
    
    def get_tool():
        wikipedia_tool = WikipediaQueryRun(api_wrapper = Wikipedia.__wikipedia_api)
        wikipedia_tool = wikipedia_tool
        return wikipedia_tool
        
        

