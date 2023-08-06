import os
import numpy as np

def image_mask_generator(data_dir, shuffle=True):
    patient_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    
    while True:
        if shuffle:
            np.random.shuffle(patient_folders)
        
        for patient_folder in patient_folders:
            image_files = [file for file in os.listdir(os.path.join(data_dir, patient_folder)) if file.startswith("image")]
            mask_files = [file for file in os.listdir(os.path.join(data_dir, patient_folder)) if file.startswith("mask")]
            
            for i in range(min(len(image_files), len(mask_files))):
                image_path = os.path.join(data_dir, patient_folder, image_files[i])
                mask_path = os.path.join(data_dir, patient_folder, mask_files[i])
                
                image = np.load(image_path)
                mask = np.load(mask_path)
                
                yield image, mask

# Example usage
data_dir = "data"  # Replace with the path to the directory containing patient folders
generator = image_mask_generator(data_dir, shuffle=True)

# Fetch a pair of images from the generator
image, mask = next(generator)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
