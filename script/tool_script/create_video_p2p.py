import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import random
from pathlib import Path

    
def process(MODEL: str, COMBINE: bool, FRAME: int = 100) -> None:
    # PATH = "/root/mmgeneration/tests/haomo/%s/%s" % (DEMO, MODEL)
    # PATH = "/mnt/ve_share/generation/data/%s/%s" % (DEMO, MODEL)
    print(MODEL)
    FATHER_PATH = "/mnt/ve_share/generation/data/p2p/imgs/%s" % MODEL
    img_paths = glob.glob(os.path.join(FATHER_PATH,'*'))
    img_paths = [_.split("_") for _ in img_paths]
    
    ft1, ft2, ft3, ft4, ft0_rf, ft0_rp = [], [], [], [], [], []
    for each in img_paths:
        if MODEL == "replace_blend_reweight" and each[-2].split("/")[-1] == "snowy":
            ft0_rp.append("_".join(each))
        elif MODEL == "refine":
            if len(each) == 4:
                ft1.append("_".join(each))
            elif len(each) == 3:
                ft0_rf.append("_".join(each))

            else:
                if each[-2] == "2":
                    ft2.append("_".join(each))
                elif each[-2] == "3":
                    ft3.append("_".join(each))
                elif each[-2] == "4":
                    ft4.append("_".join(each))
                
    fts = [ft1, ft2, ft3, ft4, ft0_rf, ft0_rp]
    fts = [random.sample(_, min(FRAME, len(_))) for _ in fts]
    
    if COMBINE:
        for i, ft in tqdm(enumerate(fts)):
            gif_images_ori = []
            for f in ft:
                paths = "%s/0.80_0.80_2.00" % (f)
                if Path(paths).exists():
                    img_ppp = glob.glob(os.path.join(paths,'*.png'))
                    print(img_ppp)
                    if "snow" in img_ppp[0].split("/")[-1]:
                        img_combine = cv2.hconcat([cv2.imread(img_ppp[1]), cv2.imread(img_ppp[0])])
                    else:
                        img_combine = cv2.hconcat([cv2.imread(img_ppp[0]), cv2.imread(img_ppp[1])])
                    
                gif_images_ori.append(Image.fromarray(np.uint8(cv2.cvtColor(img_combine, cv2.COLOR_BGR2RGB) )))
            
            save_root = "/mnt/ve_share/generation/data/result/diffusions/vis/p2p/gif/%s" % MODEL
            os.makedirs(save_root, exist_ok=True)
            save_path = "%s/%d_%d.gif" % (save_root, i, FRAME)

            # imageio.mimsave(save_path, gif_images_ori, duration=2000)
            if len(ft) > 0:
                gif_images_ori[0].save(save_path, format='GIF', append_images=gif_images_ori[1:], save_all=True, duration=1500, loop=0)
                print(save_path)
    

MODELS = ["refine", "replace_blend_reweight"]
# MODELS = ["replace_blend_reweight"]


for model in MODELS:
    process(MODEL=model, COMBINE=True, FRAME=30)