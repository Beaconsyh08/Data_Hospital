import cv2
import os
from tqdm import tqdm

# Directory containing your image files
image_folder = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_combine/ori"

fps = 2
output_root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/mp4s/%d" % fps
os.makedirs(output_root, exist_ok=True)

# Make sure the images are in the correct order
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


# Define the output video file
video_name = '%s/%s.mp4' % (output_root, "ori")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

for image in tqdm(images):
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (width, height))
    out.write(frame)

out.release()
print(video_name)