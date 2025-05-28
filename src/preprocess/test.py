import os
import json

with open("data/Linux/group_list.json",'r',encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for item in data:
    name = item["name"]
    desc = item["descroption"]
    new_item = {
        "name":name,
        "description":desc
    }
    new_data.append(new_item)

with open("data/Linux/group_list_v2.json", "w") as f:
        json.dump(new_data, f, indent=4)