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
from dotenv import load_dotenv

# Add the following import in app.py
from babelcloud_rgb import BabelCloudRGB
import traceback

load_dotenv()


INDEX = 1
@st.cache_resource
def preprocessing():
    tool = TavilySearch(max_results = 10)
    
    # Modify template to strongly encourage showing reasoning
    template = """
            Answer the following questions as best you can. 
            ALWAYS show your detailed step-by-step reasoning for every question.
            Break down each problem into smaller steps and show how you solve each step.
            You must be able to reasonably explain why the topic you searched was related to the user question.
            You have access to the following tools:

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
        template = template,
        input_variables = ["input", "agent_scratchpad"],
        partial_variables = {"tool_names": tool.name, "tools": [tool]}
    )
    
    # Rest of your code
    sources = ["HuggingFace", "Anthropic", "OpenAI"]
    llm = LLM(source=sources[INDEX])
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
reflect_and_react = build_self_reflective_agent(llm, tool, prompt, max_reflection_turns=3)

st.title("Reasoning AI")

with st.sidebar:
    models = st.selectbox("Models Available", ["HuggingFace", "Anthropic", "OpenAI"], index=INDEX)
    
    st.markdown("## 📊 Evaluate Agent on LLM-RGB")
    
    # Add test case quantity selection
    num_test_cases = st.slider("Number of test cases", min_value=1, max_value=20, value=5, 
                            help="Select how many random test cases to evaluate")
    
    if st.button("Run RGB Evaluation", key="rgb_eval_btn"):
        with st.spinner(f"Running evaluation on {num_test_cases} random test cases..."):
            try:
                # Use user-selected quantity for random sampling
                data = evaluate_rgb.load_llm_rgb_testcases(random_sample=num_test_cases)
                # Evaluate all loaded test cases
                acc, wrongs = evaluate_rgb.evaluate_agent_on_testcases(data, max_samples=None)
                
                st.success(f"✅ Accuracy: {acc:.2%}")

                # Display generated chart
                st.image("accuracy_pie_chart.png", caption="Accuracy Breakdown")
                
                # If reflection rounds distribution chart was generated, display it
                if os.path.exists("reflection_rounds_distribution.png"):
                    st.image("reflection_rounds_distribution.png", caption="Reflection Rounds Distribution")
                
                # If reasoning depth accuracy chart was generated, display it
                if os.path.exists("accuracy_by_reasoning_depth.png"):
                    st.image("accuracy_by_reasoning_depth.png", caption="Accuracy by Reasoning Depth")
                
                # Provide link to HTML report
                if os.path.exists("reflection_eval_report.html"):
                    st.markdown("[📄 View Detailed Evaluation Report](reflection_eval_report.html)")

                if wrongs:
                    st.markdown("### ❌ Wrong Predictions")
                    for i, item in enumerate(wrongs):
                        with st.expander(f"Example {i+1}"):
                            st.markdown(f"""
                            **ID:** {item.get('id', 'Unknown')}  
                            **Question:** {item['question'][:100]}...  
                            **Expected Answer:** {item['expected']}  
                            **Agent Output:** {item['output'][:150]}...  
                            **Reflection Rounds:** {item['reflection_rounds']}  
                            **Difficulty Score:** {item.get('difficulty_score', 'N/A')}  
                            """)
                            with st.expander("Full Reflection"):
                                st.text(item['reflection'])
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

    st.markdown("---")
    st.markdown("## 🧠 Evaluate Self-Reflection Agent")
    st.markdown("Evaluate your self-reflection agent implementation using the LLM-RGB benchmark")
    
    # Select number of test cases
    test_count = st.slider("Number of test cases", min_value=5, max_value=15, value=10)
    
    # Evaluation button
    if st.button("Evaluate Self-Reflection Agent", key="eval_reflection_agent"):
        with st.spinner("Evaluating self-reflection agent..."):
            try:
                # Initialize LLM-RGB evaluator, pass in self-reflection agent
                evaluator = BabelCloudRGB(agent=reflect_and_react)
                
                # Run evaluation on the agent
                experiment_dir = evaluator.run_agent_evaluation()
                
                if experiment_dir:
                    # Parse results
                    results = evaluator.parse_results(experiment_dir)
                    
                    # Display results
                    st.success("Evaluation complete!")
                    
                    # Create performance overview
                    st.subheader("Self-Reflection Agent Performance Overview")
                    
                    # Adjust this according to actual result format
                    metrics = {
                        "Reasoning Depth": results.get("reasoning_depth", 0),
                        "Self-Reflection Count": results.get("reflection_rounds", 0),
                        "Accuracy": results.get("accuracy", 0),
                        "Total Score": results.get("total_score", 0)
                    }
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    for i, (metric, value) in enumerate(metrics.items()):
                        if i % 2 == 0:
                            col1.metric(metric, f"{value:.2f}")
                        else:
                            col2.metric(metric, f"{value:.2f}")
                    
                    # Display detailed evaluation results
                    st.subheader("Test Case Detailed Results")
                    for i, test_case in enumerate(results.get("test_cases", [])):
                        with st.expander(f"Test Case {i+1}: {test_case.get('name', 'Unnamed')}"):
                            st.markdown(f"**Prompt**: {test_case.get('prompt', '')}")
                            st.markdown(f"**Expected Answer**: {test_case.get('expected', '')}")
                            st.markdown(f"**Agent Output**: {test_case.get('output', '')}")
                            st.markdown(f"**Reflection Process**: {test_case.get('reflection', '')}")
                            st.markdown(f"**Score**: {test_case.get('score', 0)}/100")
                else:
                    st.error("Evaluation did not return valid results")
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                st.error(f"Detailed error: {traceback.format_exc()}")

                
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
        resp = llm.invoke(user_prompt)
        with st.chat_message("Base Model"):
            st.write(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})

    with st.container():
        st.subheader("Out of Box ReAct Agent")
        with st.chat_message("user"):
            st.write(user_prompt)
        resp = agent.invoke({"input": user_prompt})
        with st.chat_message("ReAct Agent"):
            st.write(resp.get("output"))
            with st.expander("Reasoning Steps:"):
                st.write(resp.get("intermediate_steps"))
        st.session_state.messages.append({"role": "assistant", "content": resp.get("output")})

        #############################
        # 添加这些调试代码 DEBUG code
        # st.write("Debug - Raw response object type:")
        # st.write(type(resp))
        # st.write("Debug - Response keys:")
        # st.write(list(resp.keys()))
        # st.write("Debug - Raw intermediate steps:")
        # st.write(resp.get("intermediate_steps"))

        # with st.chat_message("ReAct Agent"):
        #     st.write(resp.get("output"))
        #     with st.expander("Reasoning Steps:"):
        #         st.write(resp.get("intermediate_steps"))
        # st.session_state.messages.append({"role": "assistant", "content": resp.get("output")})




    with st.container():
        st.subheader("🧠 Self-Reflective Agent")

        final_response, final_reflection, final_trace, all_reflections = reflect_and_react(
            user_prompt, collect_reflections=True
        )

        ####################################
        # 添加这些调试代码 DEBUGcode
        # st.write("Debug - Final response type:")
        # st.write(type(final_response))
        # st.write("Debug - Final response keys:")
        # st.write(list(final_response.keys() if hasattr(final_response, 'keys') else []))
        # st.write("Debug - Final trace type and length:")
        # st.write(type(final_trace))
        # st.write(len(final_trace) if isinstance(final_trace, str) else "Not a string")
        # st.write("Debug - All reflections count:")
        # st.write(len(all_reflections) if all_reflections else 0)
        
        # with st.chat_message("🧠 Self-Reflective Agent"):
        #     st.markdown("**🟡 Final Answer:**")
        #     st.markdown(final_response.get("output"))





        with st.chat_message("🧠 Self-Reflective Agent"):
            st.markdown("**🟡 Final Answer:**")
            st.markdown(final_response.get("output"))

        with st.expander("🔍 Reasoning Trace (Final Round)"):
            st.markdown(f"```text\n{final_trace}\n```")


        with st.expander("🪞 Reflection Rounds"):
            for r in all_reflections:
                st.markdown(f"### 🔁 Round {r['attempt']}")
                st.markdown("**Trace:**")
                st.markdown(f"```text\n{r['trace']}\n```")
                st.markdown("**Reflection:**")
                st.markdown(r["reflection"])
                st.markdown("---")  # Add separator between different rounds

        if hasattr(agent, "memory") and agent.memory is not None:
            with st.expander("🧠 Conversation Memory"):
                history_text = agent.memory.buffer_as_str()
                st.markdown(f"```text\n{history_text}\n```")





__all__ = ["reflect_and_react"]