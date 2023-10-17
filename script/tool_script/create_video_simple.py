import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image



ROOT_PATH = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_combine/make_it_night_0.4.2"
save_path = "/mnt/share_disk/leiyayun/data/hozon_neta/result/mp4s/night.gif"
    
    
def process(ROOT_PATH: str,) -> None:
    img_paths = glob.glob(os.path.join(ROOT_PATH,'*.png'))
    img_paths = sorted(img_paths)
    
    gif_images_ori, gif_images_low = [], []
    
    
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        # gif_images_ori.append(Image.open(path).resize((1024, 576)))
        gif_images_ori.append(Image.open(path))
        
    gif_images_ori[0].save(save_path, format='GIF', append_images=gif_images_ori[1:], save_all=True, duration=500, loop=0)
    print(save_path)
    

process(ROOT_PATH=ROOT_PATH)