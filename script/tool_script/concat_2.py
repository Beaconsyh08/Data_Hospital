import cv2
import os
from tqdm import tqdm

# Directory containing your image files
root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_cn_combine"
root_2 = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_combine"
res_root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_cn_vs/"

for scene in os.listdir(root):
    image_folder = '%s/%s' % (root, scene)
    image_folder_2 = '%s/%s' % (root_2, scene)
    output_root = "%s/%s" % (res_root, scene)
    os.makedirs(output_root, exist_ok=True)
    
    # Make sure the images are in the correct order
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    images2 = [img for img in os.listdir(image_folder_2) if img.endswith(".png")]
    images2.sort()
    
    print(output_root)

    for i in tqdm(range(len(images))):
        img_path_1 = os.path.join(image_folder, images[i])
        img_path_2 = os.path.join(image_folder_2, images2[i])
        frame_1 = cv2.imread(img_path_1)
        frame_2 = cv2.imread(img_path_2)
        frame = cv2.hconcat([frame_1, frame_2])
        # frame = cv2.vconcat([frame_1, frame_2])
        cv2.imwrite("%s/%s" % (output_root, images[i]), frame)
        