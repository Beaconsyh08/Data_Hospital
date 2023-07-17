import os
from tqdm import tqdm 

# Define the source directory where the original folders are located
source_directory = '/mnt/share_disk/syh/data/imgs/replace_blend_reweight'

# Define the destination directory where the folders will be moved
destination_directory = '/mnt/share_disk/syh/data/p2p/imgs/replace_blend_reweight'

# Iterate over the folders in the source directory
for folder_name in tqdm(os.listdir(source_directory)):
    # Split the folder name into A and B
    a, b = folder_name.split('_')
    
    
    # Create the destination folder path for A
    destination_a = os.path.join(destination_directory, a)
    
    # Create the destination folder path for B
    destination_b = os.path.join(destination_a, b)
    
    
    # Create the destination folder for A if it doesn't exist
    # if not os.path.exists(destination_a):
    #     os.makedirs(destination_a)
    
    # # Create the destination folder for B inside A if it doesn't exist
    # if not os.path.exists(destination_b):
    #     os.makedirs(destination_b)
    
    # # Move the contents of the original folder to the destination folder for B
    source_folder = os.path.join(source_directory, folder_name)
    # print(source_folder, destination_b)
    
    os.system('mv {} {}'.format(source_folder, destination_b))
    
    # for file_name in os.listdir(source_folder):
    #     source_file = os.path.join(source_folder, file_name)
    #     destination_file = os.path.join(destination_b, file_name)
        # shutil.move(source_file, destination_file)
    
    # Optionally, remove the original folder after moving the contents
    # shutil.rmtree(source_folder)
