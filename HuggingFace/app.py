import streamlit as st
from llm import LLM
from react_agent import ReactAgent
import torch, os
from dotenv import load_dotenv

load_dotenv()

torch.classes.__path__ = [] # add this line to manually set it to empty. 

INDEX = 1
@st.cache_resource
def preprocessing():
    llm = LLM().llm
    agent = ReactAgent()

    return llm, agent


llm, agent = preprocessing()


st.title("Reasoning AI")

with st.sidebar:
    st.markdown("## 📊 Evaluate Agent on LLM-RGB")
    if st.button("Run RGB Evaluation (10 samples)", key="rgb_eval_btn"):
        with st.spinner("Running evaluation..."):
            try:
                st.write("Coming Soon!")
                # data = evaluate_rgb.load_rgb_dataset()
                # acc, wrongs = evaluate_rgb.evaluate_agent_on_rgb(data, max_samples=10)
                # st.success(f"✅ Accuracy: {acc:.2%}")

                # st.image("accuracy_pie_chart.png", caption="Accuracy Breakdown")

                # if wrongs:
                #     st.markdown("### ❌ Wrong Predictions")
                #     for i, item in enumerate(wrongs):
                #         with st.expander(f"Example {i+1}"):
                #             st.markdown(f"""
                #             **Question:** {item['question']}  
                #             **Expected Answer:** {item['expected']}  
                #             **Agent Output:** {item['output']}  
                #             **Reflection:** {item['reflection']}  
                #             """)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Enter your question here..."):
    # Display user message in chat message container

    with st.container():
        st.subheader("Base Model")
        with st.chat_message("user"):
            st.write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        resp = llm(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
        with st.chat_message("Base Model"):
            st.write(resp.content)
    with st.container():
        st.subheader("Out of Box ReAct Agent")
        with st.chat_message("user"):
            st.write(user_prompt)
        
        resp = agent.run(user_prompt)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("ReAct Agent"):
            st.write(resp)
            with st.expander("Reasoning Steps:"):
                st.write(agent.get_memory())