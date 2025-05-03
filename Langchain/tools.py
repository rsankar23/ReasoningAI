from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field


api_wrapper = WikipediaAPIWrapper(top_k_results = 5)
WikiTool = WikipediaQueryRun(api_wrapper = api_wrapper,)
