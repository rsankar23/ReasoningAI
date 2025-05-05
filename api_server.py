from flask import Flask, request, jsonify
import os
from reflection import build_self_reflective_agent
from llm import LLM
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 初始化代理
def setup_agent():
    tool = TavilySearch(max_results=10)
    
    template = """
    Answer the following questions as best you can. Do not search unrelated topics beyond the scope of the user query. You must be able to reasonably explain why the topic you searched was related to the user question. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"tool_names": tool.name, "tools": [tool]}
    )
    
    # 使用您当前的默认模型
    llm = LLM(source="Anthropic")
    
    return build_self_reflective_agent(llm, tool, prompt, max_reflection_turns=3)

# 初始化代理
agent = setup_agent()

@app.route('/api/reason', methods=['POST'])
def reason():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # 调用自反思代理
        response, reflection, trace, all_reflections = agent(query, collect_reflections=True)
        
        # 返回结果
        return jsonify({
            "answer": response.get("output"),
            "reasoning_trace": trace,
            "reflections": [{"attempt": r["attempt"], "reflection": r["reflection"]} for r in all_reflections]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)