import json
from tqdm import tqdm
from app import reflect_and_react


def load_rgb_dataset(path="datasets/information_integration/test.json"):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_agent_on_rgb(task_data, max_samples=10):
    print(f"Evaluating on {min(max_samples, len(task_data))} samples...")
    correct = 0
    total = min(len(task_data), max_samples)

    for i in tqdm(range(total)):
        item = task_data[i]
        question = item["question"]
        context = item["context"]
        expected = item["answer"].strip().lower()

        input_text = f"{context}\n\nQuestion: {question}"
        response, reflection, trace = reflect_and_react(input_text)
        output = response.get("output", "").lower()

        if expected in output:
            correct += 1

    accuracy = correct / total
    print(f"✅ Accuracy on RGB subset: {accuracy:.2%} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    data = load_rgb_dataset()
    evaluate_agent_on_rgb(data, max_samples=10)
