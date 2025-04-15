from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os




class LLM:
    def __init__(self, repo_id = "microsoft/Phi-3-mini-4k-instruct"):
        self.endpoint = HuggingFaceEndpoint(
            repo_id = repo_id,
            task = "text-generation",
            do_sample = True,
            repetition_penalty = 1.03,
            api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            #
        )
        self.llm = ChatHuggingFace(
            llm = self.endpoint,
            verbose=True
        )
        self.history = []
        self.sys_prompt = ("system","""You are an advanced conversational AI that provides clear, concise, and helpful responses. Maintain a friendly and engaging tone throughout the conversation. If the user's question is ambiguous or unclear, ask for clarification. Always provide accurate information based on the knowledge you have, and when necessary, explain your reasoning or provide context to help the user understand the response. Avoid overwhelming the user with excessive detail unless specifically asked for. DO NOT INCLUDE THE USER QUERY IN YOUR RESPONSE - SIMPLY ANSWER THE QUESTION.
                            When replying, follow these guidelines:
                            - Be polite, empathetic, and patient.
                            - Use simple language to ensure clarity.
                            - Provide answers that are actionable, informative, or provide further helpful guidance.
                            - If the answer requires an opinion, explain your thought process.
                            - Be honest about what you do not know or cannot answer.""")

    def add_to_history(self, message):
        self.history.append(message)

    def invoke(self, query):
        temp_question = [self.sys_prompt, ("human", query)]
        return self.llm.invoke(temp_question)
