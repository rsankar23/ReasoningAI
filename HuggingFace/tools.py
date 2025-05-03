from smolagents import DuckDuckGoSearchTool, FinalAnswerTool
from dotenv import load_dotenv

load_dotenv()
class Tools:
    def __init__(self):
        self.duck_tool = DuckDuckGoSearchTool(max_results=15)
        self.final_tool = FinalAnswerTool()


    def get_search_tool(self):
        return self.duck_tool
    
    def get_final_tools(self):
        return self.final_tool

    def get_tools(self):
        return [self.duck_tool, self.final_tool]