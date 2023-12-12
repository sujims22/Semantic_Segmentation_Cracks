import os
import shutil
import numpy as np

def split_data(source_folder_images, source_folder_masks, dest_folder, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits the dataset into training, validation, and testing sets.

    :param source_folder_images: The folder path for the images.
    :param source_folder_masks: The folder path for the masks.
    :param dest_folder: The destination folder to save the splits.
    :param split_ratio: A tuple indicating the split ratio for (train, val, test).
    """
    # Ensure the destination folders exist
    sets = ['train', 'val', 'test']
    for set_name in sets:
        os.makedirs(os.path.join(dest_folder, set_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, set_name, 'masks'), exist_ok=True)

    # Get all filenames (assuming image filenames without extension are the same as mask filenames)
    all_filenames = [os.path.splitext(f)[0] for f in os.listdir(source_folder_images) if f.endswith('.jpg')]
    np.random.shuffle(all_filenames)  # Shuffle to ensure random split

    # Calculate split indices
    train_end = int(len(all_filenames) * split_ratio[0])
    val_end = train_end + int(len(all_filenames) * split_ratio[1])

    # Split filenames
    train_files = all_filenames[:train_end]
    val_files = all_filenames[train_end:val_end]
    test_files = all_filenames[val_end:]

    # Copy files into respective folders
    for set_name, files in zip(sets, [train_files, val_files, test_files]):
        for filename in files:
            image_path = os.path.join(source_folder_images, filename + '.jpg')
            mask_path = os.path.join(source_folder_masks, filename + '.png')

            if os.path.exists(image_path) and os.path.exists(mask_path):
                shutil.copy(image_path, os.path.join(dest_folder, set_name, 'images', filename + '.jpg'))
                shutil.copy(mask_path, os.path.join(dest_folder, set_name, 'masks', filename + '.png'))
            else:
                print(f"File missing: Image: {filename}.jpg or Mask: {filename}.png")

# Example usage
source_folder_images = r"/Users/sujitharavichandran/Documents/GitHub/Semantic_Segmentation_Cracks/Damage dataset/imageDamage/imageDamage"  # Update with the path to your images folder
source_folder_masks = r"/Users/sujitharavichandran/Documents/GitHub/Semantic_Segmentation_Cracks/Damage dataset/imageDamage/CoarseDamagePNG"  # Update with the path to your masks folder
dest_folder = r"/Users/sujitharavichandran/Documents/GitHub/Semantic_Segmentation_Cracks/Damage dataset/imageDamage/CoarseDamage_split"  # Update with your desired destination path

split_data(source_folder_images, source_folder_masks, dest_folder)
