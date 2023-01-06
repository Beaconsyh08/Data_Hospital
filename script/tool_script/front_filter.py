import json
from tqdm import tqdm
import threading


json_path = "/data_path/side_night.txt"
save_path = "/data_path/front_night.txt"

json_set = set()
with open(json_path) as input_file:
    json_ps = [_.strip() for _ in input_file]
    
for json_p in tqdm(json_ps):
    with open(json_p.strip()) as json_obj:
        json_info = json.load(json_obj)
        datas = json_info["relative_sensors_data"]
        for data in datas:
            if data.get("camera_orientation") == "front_middle_camera":
                json_set.add(data["image_json"])
                

# t0 = threading.Thread(target=visualizer.draw_bbox_0, args=(obs_list, save_path))
# t0.start()

with open(save_path, "w") as output_file:
    for json_ in json_set:
        output_file.writelines(json_ + "\n")
        