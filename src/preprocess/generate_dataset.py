import json
import openai
from openai import OpenAI
from tqdm import tqdm
import time

def gpt_completion(prompt):
    client = OpenAI(
        api_key="sk-BwTI1iSg83soUQ6u2d1096B8A27848E5B3E4141154Dc592b",  # 填写上api-key
        base_url="https://api.yesapikey.com/v1"
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4o-2024-05-13",
                    temperature=0   
            )
            flag = True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

if __name__=="__main__":
    dataset = []
    with open("data/Linux/group_list.json", "r") as f:
        data = json.load(f)
    with open("prompt/linux_prompt.txt","r") as f:
        prompt_template = f.read()
    for item in tqdm(data, desc="Generating Dataset", unit="item"):  # 添加 tqdm 进度条
        name = item["name"]
        desc = item["description"]
        prompt = prompt_template.format(name=name, description=desc)
        req = gpt_completion(prompt)
        dataset_item = {
            "requirement": req,
            "artifact": name,
        }
        dataset.append(dataset_item)
    with open("data/Linux/dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)