import json
import pandas as pd 
import os


ROOT = "/root/cmh/evaluation_1024_res"

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
        result = json_obj["0-60m"]
        vehicle = result["vehicle"]
        vru = result["vru"]
        
        res_dict["vehicle_precision_bev"] = vehicle["precision_bev"]
        res_dict["vehicle_recall_bev"] = vehicle["recall_bev"]
        res_dict["vehicle_f1_score_bev"] = vehicle["f1_score_bev"]
        
        res_dict["vru_precision_bev"] = vru["precision_bev"]
        res_dict["vru_recall_bev"] = vru["recall_bev"]
        res_dict["vru_f1_score_bev"] = vru["f1_score_bev"]
        
        res_dict["vehicle_precision"] = vehicle["precision"]
        res_dict["vehicle_recall"] = vehicle["recall"]
        res_dict["vehicle_f1_score"] = vehicle["f1_score"]
        
        res_dict["vru_precision"] = vru["precision"]
        res_dict["vru_recall"] = vru["recall"]
        res_dict["vru_f1_score"] = vru["f1_score"]
        
        combined_lst.append(res_dict)
        

df = pd.DataFrame(combined_lst)
df = df.set_index("name")
df = df.sort_values("name")
df.T.to_excel("%s/result_2d_3d.xlsx" % ROOT)
