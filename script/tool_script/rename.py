import json
import os
from tqdm import tqdm

ROOT = "/root/data_hospital_data/0728v60/traffic_temp/inference_doctor/dt_old/"
SAVE_ROOT = "/root/data_hospital_data/0728v60/traffic_temp/inference_doctor/dt/"
os.makedirs(SAVE_ROOT, exist_ok=True)

file_lst = []
for root, _, files in os.walk(ROOT):
    for file in files:
        file_name = os.path.join(root, file)
        file_lst.append(file_name)
            
print(len(file_lst))

for idx, json_file in tqdm(enumerate(file_lst), total=len(file_lst)):
    with open(json_file) as json_obj:
        json_info = json.load(json_obj)
        timestamp = json_info["timestamp"]
        
        save_path = "%s%d_%d.json" % (SAVE_ROOT, 0, timestamp)
        with open(save_path, 'w') as output_file:
            json.dump(json_info, output_file)