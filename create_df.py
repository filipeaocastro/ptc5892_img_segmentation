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
    
    expand_dimentions(images_list)
    expand_dimentions(masks_list)
    data = {
        'patient_id': patient_ids,
        'image': images_list,
        'mask': masks_list
    }
    df = pd.DataFrame(data)
    
    return df

def expand_dimentions(img_list):
    for i, img in enumerate(img_list):
        img_list[i] = np.expand_dims(img, axis=-1)
    

def crop_center_image(image, crop_size):
    height, width = image.shape[:2]
    start_y = max(0, height // 2 - crop_size // 2)
    end_y = min(height, height // 2 + crop_size // 2)
    start_x = max(0, width // 2 - crop_size // 2)
    end_x = min(width, width // 2 + crop_size // 2)
    
    cropped_image = image[start_y:end_y, start_x:end_x, :]
    return cropped_image

def crop_center_images_in_dataframe(df, crop_size=168):
    cropped_images = []
    cropped_masks = []
    
    for index, row in df.iterrows():
        image = row['image']
        mask = row['mask']
        
        cropped_image = crop_center_image(image, crop_size)
        cropped_mask = crop_center_image(mask, crop_size)
        
        cropped_images.append(cropped_image)
        cropped_masks.append(cropped_mask)

    for i, (image, mask) in enumerate(zip(cropped_images, cropped_masks)):
        df['image'][i] = image
        df['mask'][i] = mask

    # return df


crop = True

if not crop:
    data_dir = "data" 
    df = create_dataframe(data_dir)

    with open('imgs.pkl', 'wb') as f:
        pickle.dump(df, f)
else:
    with open('imgs.pkl', 'rb') as f:
        df = pickle.load(f)
    crop_center_images_in_dataframe(df, crop_size=128)
    with open('imgs_cropped.pkl', 'wb') as fw:
        pickle.dump(df, fw)
    


    

