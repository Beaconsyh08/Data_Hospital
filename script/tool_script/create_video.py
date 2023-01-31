import imageio
import glob
import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import cv2

def process(MODEL: str, DEMO: str, COMBINE: bool, FRAME: int = 100) -> None:
    # PATH = "/root/mmgeneration/tests/haomo/%s/%s" % (DEMO, MODEL)
    # PATH = "/mnt/ve_share/generation/data/%s/%s" % (DEMO, MODEL)
    
    PATH = "/mnt/ve_share/generation/data/result/%s/%s_latest/images/fake_B" % (MODEL, DEMO)
    img_paths = glob.glob(os.path.join(PATH,'*.png'))
    img_paths = sorted(img_paths)[:FRAME]
    
    if COMBINE:
        ORI_PATH = "/mnt/ve_share/generation/data/result/%s/%s_latest/images/real_A" % (MODEL, DEMO)
        ori_img_paths = glob.glob(os.path.join(ORI_PATH,'*.png'))
        ori_img_paths = sorted(ori_img_paths)[:FRAME]

    # print(img_paths)
    gif_images_ori, gif_images_low = [], []
    
    
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        if COMBINE:
            im_combine = cv2.hconcat([imageio.imread(ori_img_paths[i]), imageio.imread(path)])
            gif_images_ori.append(im_combine)
            # gif_images_low.append(resize(im_combine,(1080//6, 1920//3)))
            gif_images_low.append(resize(im_combine,(1080//4, 1920//2)))
            
        else:
            gif_images_ori.append(imageio.imread(path))
            gif_images_low.append(resize(imageio.imread(path),(1080//6, 1920//3)))
            
    # save_path = "/root/mmgeneration/tests/haomo/video/ori/%s_%d_ORI_%s.gif" % (SCENE, FRAME, MODEL)
    # save_path = "/mnt/ve_share/generation/data/video/ori/%s/%s_%s_%d_ORI_%s.gif" % (DEMO, DEMO, SCENE, FRAME, MODEL)

    # imageio.mimsave(save_path, gif_images_ori, fps=10)
    # print(save_path)

    # save_path = "/root/mmgeneration/tests/haomo/video/low/%s/%s_%s_%d_LOW_%s.gif" % (DEMO, DEMO, SCENE, FRAME, MODEL)
    save_root = "/mnt/ve_share/generation/data/video/low/%s" % DEMO
    os.makedirs(save_root, exist_ok=True)
    save_path = "%s/%s_%d_LOW_%s.gif" % (save_root, DEMO, FRAME, MODEL)

    imageio.mimsave(save_path, gif_images_low, fps=10)
    from pygifsicle import optimize
    optimize(save_path) # For overwriting the original one
    print(save_path)
    
# "SNOW_BDDCADC", "SNOW_BDDWATERLOO", "SNOW_BDDWATERLOOHAK", "SNOW_HAK"
MODELS = ["cut_snow_hok", "cut_snow_waterloo"       ,]  #  "cut_snow_hok", "cut_snow_waterloo"
for model in MODELS:
    print(model)
    process(MODEL=model, DEMO="demo3", COMBINE=True, FRAME=100)