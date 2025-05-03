from smolagents import HfApiModel, FinalAnswerTool
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self):
        self.llm = HfApiModel(
            max_tokens=7800,
            temperature=0.5,
            model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
            custom_role_conversions=None,
        )
    def get_llm(self):
        return self.llm