import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import random
from pathlib import Path

    
def process(MODEL: str, COMBINE: bool, FRAME: int, SCENE: str) -> None:
    # PATH = "/root/mmgeneration/tests/haomo/%s/%s" % (DEMO, MODEL)
    # PATH = "/mnt/ve_share/songyuhao/generation/data/%s/%s" % (DEMO, MODEL)
    print("MODEL:", MODEL)
    print("SCENE:", SCENE)
    FATHER_PATH = "/mnt/share_disk/syh/data/prompt_to_prompt/imgs/%s/%s" % (MODEL, SCENE)
    folder_paths = glob.glob(os.path.join(FATHER_PATH,'*'))
    folder_paths = random.sample(folder_paths, min(FRAME, len(folder_paths)))
    
    if COMBINE:
        gif_images_ori = []
        for f in folder_paths:
            paths = "%s/0.80_0.80_2.00" % (f)
            if Path(paths).exists():
                img_ppp = glob.glob(os.path.join(paths,'*.png'))
                if SCENE[:-1] in img_ppp[0].split("/")[-1]:
                    img_combine = cv2.hconcat([cv2.imread(img_ppp[1]), cv2.imread(img_ppp[0])])
                else:
                    img_combine = cv2.hconcat([cv2.imread(img_ppp[0]), cv2.imread(img_ppp[1])])
                
            gif_images_ori.append(Image.fromarray(np.uint8(cv2.cvtColor(img_combine, cv2.COLOR_BGR2RGB) )))
        
        save_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/p2p/gif_3/%s" % MODEL
        os.makedirs(save_root, exist_ok=True)
        save_path = "%s/%s_%d.gif" % (save_root, SCENE, FRAME)

        # imageio.mimsave(save_path, gif_images_ori, duration=2000)
        if len(folder_paths) > 0:
            gif_images_ori[0].save(save_path, format='GIF', append_images=gif_images_ori[1:], save_all=True, duration=1500, loop=0)
            print(save_path)
    

MODELS = ["replace_blend_reweight"]
# MODELS = ["replace_blend_reweight"]


for model in MODELS:
    for SCENE in ["rainy", "snowy", "foggy", "night"]:
        process(MODEL=model, COMBINE=True, FRAME=30, SCENE=SCENE)