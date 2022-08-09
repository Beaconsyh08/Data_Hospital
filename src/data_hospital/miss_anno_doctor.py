from pathlib import Path
from configs.config import VisualizationConfig
from src.utils.struct import Obstacle, parse_obs
from src.visualization.visualization import Visualizer
import pandas as pd
import threading
import os
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
from src.utils.logger import get_logger


logger = get_logger()


def save_json(obs_list: List[Obstacle] = [], save_path:str = None):
    if obs_list == 0:
        logger.warning("List[Obstacle] == 0, cannot draw bbox")
        return
    json_path = obs_list[0].json_path
    with open(json_path) as f:
        json_obj = json.load(f)
    with open(save_path, 'w') as output_file:
        json.dump(json_obj, output_file)
    
    
def visualization(df_vis: pd.DataFrame, save_root: str):
    visualizer = Visualizer(VisualizationConfig)
    for idx, row in tqdm(enumerate(df_vis.itertuples()), total=len(df_vis)):
        obs_list = []
        class_name, idx, id = row.class_name, row.Index, row.id
        img_url = idx.split("@")[0]
        obs = parse_obs(row)
        obs_list.append(obs)
        
        save_dir = "%s/%s/%s_%d" % (save_root, class_name, str(Path(row.json_path).stem), id)
        save_path = '%s.jpg' % save_dir
        bev_save_path = '%s_bev.jpg' % save_dir
        json_save_path = '%s.json' % save_dir
        t0 = threading.Thread(target=visualizer.draw_bbox_0, args=(obs_list, save_path))
        t0.start()
        # t1 = threading.Thread(target=visualizer.plot_bird_view, args=(obs_list, bev_save_path))
        # t1.start()
        # t2 = threading.Thread(target=save_json, args=(obs_list, json_save_path))
        # t2.start()
        

if __name__ == '__main__':
    path = '/cpfs/output/0719/other/dataframes/P0_dataframe.pkl'
    df = pd.read_pickle(path)["df"]
    df_false_vru_p0 = df[(df.priority == "P0") & (df.case_flag == "2") & (df.iou == -1)]
    false_vru_p0_jsons = list(set(df_false_vru_p0.json_path))
    logger.debug("Miss Label P0 Frame Number: %d" % len(false_vru_p0_jsons))
    
    save_root = "/root/data_hospital/0728v60/sidecam_ori_2/miss_anno_doctor/"
    os.makedirs(save_root, exist_ok=True)
    
    with open("%s/to_del.txt" % save_root, "w") as to_del_file:
        with open("%s/to_del_train.txt" % save_root, "w") as to_del_train_file:
            for json_path in tqdm(false_vru_p0_jsons):
                train_json_path = ""
                with open(json_path) as f:
                    json_obj = json.load(f)
                    rel_sensors = json_obj.get("relative_sensors_data")
                    camera_orientation = json_obj.get("camera_orientation")
                    for rel_sensor in rel_sensors:
                        if rel_sensor.get("camera_orientation") == camera_orientation:
                            train_json_path = rel_sensor.get("image_json")
                            break
                
                if train_json_path != "" and train_json_path.split("/")[4] == "preparation":
                    to_del_train_file.writelines("/" + train_json_path + "\n")
                else:
                    to_del_file.writelines(json_path + "\n")
                    
    df_vis = df_false_vru_p0.sample(500)
    visualization(df_vis, save_root)

        