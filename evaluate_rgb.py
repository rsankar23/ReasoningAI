import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from reflection import build_self_reflective_agent
from llm import LLM
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate


def setup_reflect_agent():
    tool = TavilySearch(max_results=10)
    template = PromptTemplate(
        template="""
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
        """,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"tool_names": tool.name, "tools": [tool]}
    )
    llm = LLM()
    return build_self_reflective_agent(llm, tool, template)


def load_rgb_dataset(path="datasets/information_integration/test.json"):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_agent_on_rgb(task_data, max_samples=10, save_path="eval_results.csv"):
    print(f"Evaluating on {min(max_samples, len(task_data))} samples...")
    reflect_and_react = setup_reflect_agent()
    correct = 0
    total = min(len(task_data), max_samples)
    wrong_samples = []
    all_results = []

    for i in tqdm(range(total)):
        item = task_data[i]
        question = item["question"]
        context = item["context"]
        expected = item["answer"].strip().lower()

        input_text = f"{context}\n\nQuestion: {question}"
        try:
            response, reflection, trace = reflect_and_react(input_text)
            output = response.get("output", "").lower()
            is_correct = expected in output
            if is_correct:
                correct += 1
            else:
                wrong_samples.append({
                    "question": question,
                    "context": context,
                    "expected": expected,
                    "output": output,
                    "reflection": reflection
                })

            all_results.append({
                "question": question,
                "expected": expected,
                "output": output,
                "reflection": reflection,
                "correct": is_correct
            })

        except Exception as e:
            print(f"[{i+1}] Skipped due to error: {e}")

    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"📁 Saved evaluation results to {save_path}")

    # Plot pie chart
    plt.figure(figsize=(4,4))
    plt.pie([correct, total - correct], labels=["Correct", "Wrong"], autopct='%1.1f%%', colors=["#4CAF50", "#F44336"])
    plt.title("Agent Accuracy on RGB")
    plt.savefig("accuracy_pie_chart.png")
    print("📊 Saved pie chart as accuracy_pie_chart.png")

    accuracy = correct / total
    print(f"✅ Accuracy on RGB subset: {accuracy:.2%} ({correct}/{total})")
    return accuracy, wrong_samples


if __name__ == "__main__":
    data = load_rgb_dataset()
    evaluate_agent_on_rgb(data, max_samples=10)
