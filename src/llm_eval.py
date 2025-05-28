import os, sys, json, time
from tqdm import tqdm
from openai import OpenAI
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from src.compute_metrics import evaluate_all

os.environ['VLLM_USE_MODELSCOPE'] = 'True'

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def llm_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--ecosystem", type=str, default="Linux")
    parser.add_argument("--prompt", type=str, default="prompt/eval_llm/")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--output", type=str, default="output/eval_llm/")
    parser.add_argument("--metric", type=str, default="metrics/eval_llm")
    return parser.parse_args()

def deepseek_completion(prompt):
    client = OpenAI(api_key="sk-xxx", base_url="https://api.deepseek.com")
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "system", "content": "You are a helpful assistant"},
                          {"role": "user", "content": prompt}],
                stream=False)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            time.sleep(0.5)

def gpt_completion(prompt):
    client = OpenAI(api_key="sk-xxx", base_url="https://api.yesapikey.com/v1")
    while True:
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4-0613",
                temperature=0)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            time.sleep(0.5)

def ask_llm(args, model, sampling_params, requirement, description):
    prompt = f"""Given the following software requirement and the description of a software artifact, determine whether the artifact can satisfy the requirement.

Requirement:
{requirement}

Artifact Description:
{description}

Can this artifact satisfy the requirement? Answer with "Yes" or "No" only."""
    if model is None:
        if args.model in ["gpt-4", "gpt-3.5-turbo"]:
            return gpt_completion(prompt)
        elif args.model == "deepseek-r1":
            return deepseek_completion(prompt)
    else:
        result = model.generate([prompt], sampling_params)
        return result[0].outputs[0].text.strip()

def main():
    args = llm_parser_args()
    valid_data_path = f"{args.data}/{args.ecosystem}/dataset.json"
    candidate_data_path = f"{args.data}/{args.ecosystem}/candidate.json"
    dataset = load_json_file(valid_data_path)
    candidate_data = load_json_file(candidate_data_path)

    predict_info = {}
    ground_info = {}
    mp1 = {item['name']: idx for idx, item in enumerate(candidate_data)}
    processing_times = {}

    for req_id, data in enumerate(dataset[:5]):
        requirement = data['requirement']
        artifact = data['artifact']
        ground_info[req_id] = mp1[artifact]
        
        matched_ids = []
        start_time = time.time()

        for item in tqdm(candidate_data, desc=f"Requirement {req_id}"):
            name = item["name"]
            description = item["description"]
            answer = ask_llm(args, None, None, requirement, description)
            if answer.lower().strip().startswith("yes"):
                matched_ids.append(mp1[name])
            if len(matched_ids) >= 10:
                break

        predict_info[req_id] = matched_ids
        end_time = time.time()
        processing_times[req_id] = end_time - start_time

    output_path = f"{args.output}/{args.ecosystem}/{args.model}_predict.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predict_info, f, ensure_ascii=False, indent=4)

    metric1 = evaluate_all(predict_info, ground_info, 1)
    metric5 = evaluate_all(predict_info, ground_info, 5)
    metric10 = evaluate_all(predict_info, ground_info, 10)

    metric_path = f"{args.metric}/{args.ecosystem}/{args.model}_metrics.txt"
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    with open(metric_path, 'w', encoding='utf-8') as f:
        f.write(f"Metrics for {args.ecosystem} using {args.model}:\n")
        f.write(metric1 + "\n" + metric5 + "\n" + metric10 + "\n\n")
        f.write("Processing time per requirement (in seconds):\n")
        for req_id, t in processing_times.items():
            f.write(f"Requirement {req_id}: {t:.2f} seconds\n")

    print(f"Metrics saved to {metric_path}")

if __name__ == "__main__":
    main()
