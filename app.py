import streamlit as st
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from llm import LLM
from langchain_tavily import TavilySearch
# from react_agent import ReActAgent
import torch, os
from reflection import build_self_reflective_agent
import evaluate_rgb

torch.classes.__path__ = [] # add this line to manually set it to empty. 


@st.cache_resource
def preprocessing():
    tool = TavilySearch(max_results = 10)
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
    working = "You are a digital assistant tasked with responding to users questions while referencing the ultimate resource; the internet. Make the relevant focused topical searches based on the users query. Leverage the {tools} tools with the names {tool_names} at every interaction. The query is: {query}; Leverage the space here to provide your structured, reasoned thoughts, {agent_scratchpad}"    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["input", "agent_scratchpad"],
        partial_variables = {"tool_names":tool.name, "tools":[tool]}
    )
    # prompt = hub.pull("hwchase17/react")
    llm = LLM()
    agent = AgentExecutor(
        agent = create_react_agent(
            llm.llm,
            tools = [tool],
            prompt = prompt
        ),
        return_intermediate_steps = True,
            tools = [tool],
            verbose = True,
            handle_parsing_errors=True
    )
    return llm, agent, tool, prompt

llm, agent, tool, prompt = preprocessing()
reflect_and_react = build_self_reflective_agent(llm, tool, prompt)

st.title("Reasoning AI")

with st.sidebar:
    st.markdown("## 📊 Evaluate Agent on LLM-RGB")
    if st.button("Run RGB Evaluation (10 samples)"):
        with st.spinner("Running evaluation..."):
            try:
                data = evaluate_rgb.load_rgb_dataset()
                acc = evaluate_rgb.evaluate_agent_on_rgb(data, max_samples=10)
                st.success(f"✅ Accuracy: {acc:.2%}")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Enter your question here..."):
    # Display user message in chat message container

    col1, col2, col3 = st.columns(spec = ([1,2,1]), border = True, gap = "large")
    with col1:
        st.subheader("Base Model")
        with st.chat_message("user"):
            st.write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        resp = llm.invoke(user_prompt)
        with st.chat_message("Base Model"):
            st.write(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})

    with col2:
        st.subheader("Out of Box ReAct Agent")
        with st.chat_message("user"):
            st.write(user_prompt)
        resp = agent.invoke({"input": user_prompt})
        with st.chat_message("ReAct Agent"):
            st.write(resp.get("output"))
            with st.expander("Reasoning Steps:"):
                st.write(resp.get("intermediate_steps"))
        st.session_state.messages.append({"role": "assistant", "content": resp.get("output")})
    with col3:
        st.subheader("🧠 Self-Reflective Agent")
        final_response, reflection, trace = reflect_and_react(user_prompt)

        with st.chat_message("Reflective Agent"):
            st.markdown("**🟡 Final Answer:**")
            st.markdown(final_response.get("output"))

        with st.expander("🔍 Reasoning Trace"):
            st.markdown(f"```text\n{trace}\n```")

        with st.expander("🪞 Reflection"):
            st.markdown(reflection)
__all__ = ["reflect_and_react"]
