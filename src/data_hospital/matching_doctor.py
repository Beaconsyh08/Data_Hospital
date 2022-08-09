import pandas as pd    
import threading
import os 
from src.utils.struct import Obstacle, parse_obs
from pathlib import Path
from src.visualization.visualization import Visualizer
from configs.config import VisualizationConfig
from tqdm import tqdm
import json
import pickle


def save_to_pickle(pickle_obj: dict, save_path: str) -> None:
    with open(save_path, "wb") as pickle_file: 
        pickle.dump(pickle_obj, pickle_file)
            
            
if __name__ == '__main__':
    NAME = '0719'
    # path = '/cpfs/output/%s/other/dataframes/eval_dataframe.pkl' % NAME
    # df = pd.read_pickle(path)["df"]
    # # df.dropna(subset=['peer_id'])
    # print(len(df))
    
    # # with open("/data_path/v60_train.txt", "r") as v60_train_file:
    # #     v60_train = [_.strip() for _ in v60_train_file]
    
    # # PATH = "/root/data_hospital/dataframes"
    # # all_train_pkl_path = "%s/total_sidecam.pkl" % PATH
    
    # # df = df[df.json_path.isin(v60_train)]
    # # print(len(df))
    # # df_p0 = df[((df['flag'] == 'miss') & (df['case_flag']=='0')) & (df['priority'] == 'P0')]
    # df_miss = df[(df['flag'] == 'miss') & (df['case_flag'] == '0')]
    # save_to_pickle(df_miss, "/root/data_hospital/dataframes/df_miss.pkl")
    # print(len(df_miss))
    
    # # df_sorted = df_p0.sort_values("dis_ratio", ascending=False)
    
    # df_ra05_eu10 = df_miss[(df_miss.dis_ratio > 0.5) & (df_miss.euclidean_dis_diff > 5)]
    # save_to_pickle(df_ra05_eu10, "/root/data_hospital/dataframes/df_ra05_eu10.pkl")
    df_ra05_eu10 = pd.read_pickle("/root/data_hospital/dataframes/df_ra05_eu10.pkl")
    print(len(df_ra05_eu10))
    values = df_ra05_eu10.sample(100).index.to_list()

    
    visualizer = Visualizer(VisualizationConfig)
    save_root = "/root/data_hospital/0728v60/sidecam_ori_2/matching_doctor/%s" % NAME
    os.makedirs(save_root, exist_ok=True)
    
    with open("%s/to_del.txt" % save_root, "w") as to_del_file:
        with open("%s/to_del_train.txt" % save_root, "w") as to_del_train_file:
            for json_path in tqdm(set(df_ra05_eu10.json_path.to_list())):
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
        
    
    for i, idx in tqdm(enumerate(values), total=len(values)):
        row = df.loc[idx] if type(df.loc[idx]) == pd.Series else df.loc[idx].iloc[0]
        obs = parse_obs(row)
        img_name, id = Path(row.name).stem, row.id
        obs_list = [parse_obs(row)]
        dis_ratio, euclidean_dis_diff = round(row.dis_ratio, 2), round(row.euclidean_dis_diff, 2)
        
        save_dir = "%s/images/%d_%s_%s_%s" % (save_root, i, str(dis_ratio), str(euclidean_dis_diff), row.camera_orientation)
        save_path = '%s.jpg' % save_dir
        bev_save_path = '%s_bev.jpg' % save_dir
        
        if not pd.isna(row.peer_id) and row.peer_id in df.index:
            _ = df.loc[row.peer_id]
        peer_row = _ if type(_) == pd.Series else _.iloc[0]
        obs_list.append(parse_obs(peer_row))
            
        t0 = threading.Thread(target=visualizer.draw_bbox_0, args=(obs_list, save_path))
        t0.start()
        t1 = threading.Thread(target=visualizer.plot_bird_view, args=(obs_list, bev_save_path))
        t1.start()