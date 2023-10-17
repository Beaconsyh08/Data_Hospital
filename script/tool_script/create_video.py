import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image


    
def process(MODEL: str, DEMO: str, COMBINE: bool, FATHER_PATH: str, FRAME: int = 100, COMB_PATH: str = None) -> None:
    # PATH = "/root/mmgeneration/tests/haomo/%s/%s" % (DEMO, MODEL)
    # PATH = "/mnt/ve_share/songyuhao/generation/data/%s/%s" % (DEMO, MODEL)
    # FATHER_PATH = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/NORMAL_512/%s" % MODEL
    PATH = "%s/%s" %(FATHER_PATH, DEMO)
    print(PATH)
    img_paths = glob.glob(os.path.join(PATH,'*.png'))
    img_paths = sorted(img_paths)[:FRAME]
    print(len(img_paths))
    
    if COMBINE:
        COMB_PATH = "%s/%s" %(COMB_PATH, DEMO)
        combine_paths = glob.glob(os.path.join(COMB_PATH,'*.png'))
        combine_paths = sorted(combine_paths)[:FRAME]
        print(combine_paths)

    # print(img_paths)
    gif_images_ori, gif_images_low = [], []
    
    
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        if COMBINE:
            img_combine = cv2.vconcat([cv2.imread(combine_paths[i]), cv2.imread(path)])
            gif_images_ori.append(Image.fromarray(np.uint8(cv2.cvtColor(img_combine, cv2.COLOR_BGR2RGB) )))
            
            # gif_images_low.append(resize(im_combine,(1080//6, 1920//3)))
            # gif_images_low.append(resize(im_combine,(1080//4, 1920//2)))
            
        else:
        # gif_images_ori.append(imageio.imread(path))
            gif_images_ori.append(Image.open(path))
        
        # gif_images_low.append(resize(imageio.imread(path),(1080//6, 1920//3)))
            
    # save_path = "/root/mmgeneration/tests/haomo/video/ori/%s_%d_ORI_%s.gif" % (SCENE, FRAME, MODEL)
    # save_path = "/mnt/ve_share/songyuhao/generation/data/video/ori/%s/%s_%s_%d_ORI_%s.gif" % (DEMO, DEMO, SCENE, FRAME, MODEL)

    # imageio.mimsave(save_path, gif_images_ori, fps=10)
    # print(save_path)

    # save_path = "/root/mmgeneration/tests/haomo/video/low/%s/%s_%s_%d_LOW_%s.gif" % (DEMO, DEMO, SCENE, FRAME, MODEL)
    save_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/gif/%s" % MODEL
    os.makedirs(save_root, exist_ok=True)
    save_path = "%s/%s_%d.gif" % (save_root, DEMO, FRAME)

    # imageio.mimsave(save_path, gif_images_ori, duration=2000)
    gif_images_ori[0].save(save_path, format='GIF', append_images=gif_images_ori[1:], save_all=True, duration=1600, loop=0)
    # from pygifsicle import optimize
    # optimize(save_path) # For overwriting the original one
    print(save_path)
    
# "SNOW_BDDCADC", "SNOW_BDDWATERLOO", "SNOW_BDDWATERLOOHAK", "SNOW_HAK"
# MODELS = ["SD-HM-V0.1",]  #  "cut_snow_hok", "cut_snow_waterloo"
# MODELS = ["SD-HM-V0.0", "SD-HM-V0.1", "SD-Base", "SD-HM-V1.0", "SD-HM-V1.1", "SD-HM-V1.2"]
# MODELS = ["SD-Base", "SD-HM-V0.0", "SD-HM-V0.1", "SD-HM-V1.0", "SD-HM-V1.1", "SD-HM-V1.2", "SD-HM-V2.0", "SD-HM-V3.0", "SD-HM-V3.0.1", "SD-HM-V3.1", "SD-HM-V3.1.1", "SD-HM-V4.0", "SD-HM-V4.0.1", "SD-HM-V4.1", "SD-HM-V4.1.1"]
MODELS = ["INS-HM-V0.4.5-5000",]

# MODELS = ["INS-Base", "INS-HM-NIGHT-V0.0.0", "INS-HM-NIGHT-V0.0.1", "INS-HM-NIGHT-V0.1.0", "INS-HM-SNOWY-V0.0.0", "INS-HM-SNOWY-V0.0.1", "INS-HM-SNOWY-V0.1.0"]


MANUAL = False
COMBINE = False
# ROOT_PATH = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/dreambooth"
ROOT_PATH = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/official"

for model in MODELS:
    print(model)
    
    FATHER_PATH = "%s/%s" % (ROOT_PATH, model)
    COMB_PATH = "%s/%s" % (ROOT_PATH, "INS-HM-SNOWY-V0.0.0")
    print(FATHER_PATH)
    prompts = [x[0].split("/")[-1] for x in os.walk(FATHER_PATH)][1:]
    print(prompts)
    if MANUAL:
        prompts = ["a city street with cars driving down it and tall buildings in the background on a foggy day with a few cars", "a city street with cars driving down it and tall buildings in the background with a few cars", "a white bus driving down a street next to a white car and a white car with a yellow license plate", ]
        prompts = ["_".join(_.split()) for _ in prompts]
    for prompt in prompts:
        if MANUAL:
            if model == "SD-HM-V0.0.1":
                prompt += ",_in_the_style_of_haomo"
        print(prompt)
        
        # gif_path = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/gif/%s/%s_%d.gif" % (model, prompt, 21)
        # if os.path.exists(gif_path):
        #     continue
        
        process(MODEL=model, DEMO=prompt, COMBINE=COMBINE, FRAME=45, FATHER_PATH=FATHER_PATH, COMB_PATH=COMB_PATH)