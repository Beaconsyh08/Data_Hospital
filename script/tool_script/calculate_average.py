import pandas as pd
import json
import os

ROOT = "/share/analysis/result/qa_res/clean/M-PT188-139-U"

file_lst = []
for root, dirs, files in os.walk(ROOT):
    for file in files:
        file_name = os.path.join(root, file)
        if file_name.endswith(".json"):
            file_lst.append(file_name)


combined_lst = []  
for file in file_lst:
    res_dict = dict()
    with open(file, 'r') as f:
        # Modify your name rules if not match
        res_dict["name"] = file.split("/")[-1].split(".")[0].split("_")[1].capitalize()
        json_obj = json.load(f)
        result = json_obj["all_P0_bev"]
        res_dict["precision"] = result["precision"]
        res_dict["recall"] = result["recall"]
        res_dict["f1_score"] = result["f1_score"]
        
        combined_lst.append(res_dict)
        

df = pd.DataFrame(combined_lst)
df = df.set_index("name")
df = df.sort_values("name").T
df
print(df)
# df.T.to_excel("%s/result.xlsx" % ROOT)