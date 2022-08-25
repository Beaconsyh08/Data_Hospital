from pathlib import Path
from src.utils.struct import Obstacle, parse_obs
from src.visualization.visualization import Visualizer
import pandas as pd
import threading
import os
from tqdm import tqdm

def show_case(df, ):
    visualizer = Visualizer(VConfig)
    for idx, row in tqdm(enumerate(df.itertuples()), total=len(df)):
        # try:
        obs_list = []
        class_name, idx, id = row.class_name, row.Index, row.id
        img_url = idx.split("@")[0]
        obs = parse_obs(row)
        obs_list.append(obs)
        
        # if not pd.isna(row.peer_id):
        #     obs_list.append(parse_obs(df.loc[row.peer_id]))
        img_save_name = '_'.join([row.flag, row.class_name, Path(row.json_path).stem, str(int(row.id))+'.jpg'])
        
        save_path = "./cluster_img/%f_%s.png" % (round(row.yaw, 2), row.class_name)
        print(row.json_path)
        bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
        t0 = threading.Thread(target=visualizer.draw_bbox_0, args=(obs_list, save_path))
        
        # save_path = os.path.join('./cluster_img_cropped/', str(row.cluster_id), img_save_name)
        # bev_save_path = str(Path(save_path).parent/(str(Path(save_path).stem)+'_bev.jpg'))
        # t0 = threading.Thread(target=visualizer.crop_img, args=(img_url, obs_list[0].bbox, save_path))

        t0.start()
        # t1 = threading.Thread(target=visualizer.plot_bird_view, args=(obs_list, bev_save_path))
        # t1.start()
        # except Exception:
        #     continue
        
        
class VConfig():
    SAVE_DIR = '/cpfs/output/%s/other'
    MAX_TSNE_SAMPLES = 20000
    
df_path = "/root/data_hospital_data/0728v60/test0809/dataframes/reproject_dataframe.pkl"
df = pd.read_pickle(df_path)["df"]
df = df[df.yaw>3.14]
show_case(df)
