import os
import numpy as np
import pandas as pd
import nibabel as nib
import pickle

def create_dataframe(data_dir):
    patient_ids = []
    images_list = []
    masks_list = []

    dirpath = "data\\NORpatients"
    all_pats = list(os.walk(dirpath))

    for patient, (abspath, folders, files) in zip(all_pats[0][1], all_pats[1:]):
        for filename in files:
            if "_gt.npy" not in filename: continue

            path_gt = os.path.join(abspath, filename)
            gt = np.load(path_gt)
            split_f = filename.split('_gt.npy')
            img_filename = split_f[0] + ".nii.gz"
            path_img = os.path.join(abspath, img_filename)
            img = nib.load(path_img).get_fdata()

            for j in range(img.shape[2]):
                patient_ids.append(patient)
                images_list.append(img[:, :, j])
                masks_list.append(gt[:, :, j])


    #     pass
    
    # patient_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    # for patient_folder in patient_folders:
    #     image_files = [file for file in os.listdir(os.path.join(data_dir, patient_folder)) if file.startswith("image")]
    #     mask_files = [file for file in os.listdir(os.path.join(data_dir, patient_folder)) if file.startswith("mask")]
        
    #     image_indices = np.arange(len(image_files))
    #     np.random.shuffle(image_indices)
        
    #     for i in image_indices:
    #         image_path = os.path.join(data_dir, patient_folder, image_files[i])
    #         mask_path = os.path.join(data_dir, patient_folder, mask_files[i])
            
    #         image = np.load(image_path)  # Assuming the file contains shape (x, y, 10)
    #         mask = np.load(mask_path)    # Assuming the file contains shape (x, y, 10)
            
    #         for j in range(image.shape[2]):
    #             patient_ids.append(patient_folder)
    #             images_list.append(image[:, :, j])
    #             masks_list.append(mask[:, :, j])
    
    # Create the pandas DataFrame
    data = {
        'patient_id': patient_ids,
        'image': images_list,
        'mask': masks_list
    }
    df = pd.DataFrame(data)
    
    return df

# Example usage
data_dir = "data"  # Replace with the path to the directory containing patient folders
df = create_dataframe(data_dir)

with open('imgs.pkl', 'wb') as f:
    pickle.dump(df, f)
    
# Save the DataFrame to a CSV file
# df.to_csv("paired_images_masks.csv", index=True)
