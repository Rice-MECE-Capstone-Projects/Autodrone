import os
from PIL import Image


# Folder path where your files are located
folder_path = 'Dataset/test/masks_colmap'

# Get the list of file names in the folder
file_list = os.listdir(folder_path)

# Iterate through each file
for file_name in file_list:
    if file_name.endswith('_mask.png'):  # Make sure to process only mask files
        old_path = os.path.join(folder_path, file_name)
        new_name = file_name.replace('_mask.png', '.png.png')  # New file name

        # Rename the file
        os.rename(old_path, os.path.join(folder_path, new_name))

        # Invert black and white
        image = Image.open(os.path.join(folder_path, new_name))
        inverted_image = Image.eval(image, lambda x: 255 if x == 0 else 0)  # Invert black (0) and white (255)
        inverted_image.save(os.path.join(folder_path, new_name))


print("Completed file renaming and black-white inversion.")
