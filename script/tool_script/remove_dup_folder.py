import os

# Specify the path to the main folder (A)
main_folder = '/mnt/share_disk/syh/data/p2p/imgs/replace_blend_reweight/night'

# Iterate over the subfolders in the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    
    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Get the list of items (folders and files) inside the subfolder
        subfolder_items = os.listdir(subfolder_path)
        
        # Filter out the folders from the items list
        subfolder_folders = [item for item in subfolder_items if os.path.isdir(os.path.join(subfolder_path, item))]
        
        # Iterate over the folders inside the subfolder
        for folder in subfolder_folders:
            folder_path = os.path.join(subfolder_path, folder)
            
            # Check if the folder name follows the "x_y" format
            if len(folder.split("_")) == 2:
                folder_to_delete = os.path.join(subfolder_path, folder)
                # Uncomment the following line to actually delete the folder
                # os.system("rm -rf %s" % folder_to_delete)
                print("Deleted folder:", folder_to_delete)
