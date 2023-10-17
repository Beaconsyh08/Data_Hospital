import cv2
import os
from tqdm import tqdm

# Directory containing your image files
root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_combine"
for scene in os.listdir(root):
    if scene in ["make_it_night_0.4.2"]:
        image_folder = '%s/%s' % (root, scene)
        fps = 2
        output_root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/mp4s/%d" % fps
        os.makedirs(output_root, exist_ok=True)
        
        # Make sure the images are in the correct order
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()
        images = images[0:50]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # Define the output video file
        video_name = '%s/%s_.mp4' % (output_root, scene)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        for image in tqdm(images):
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            out.write(frame)

        out.release()
        print(video_name)