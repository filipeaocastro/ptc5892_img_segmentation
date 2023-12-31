import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import pickle

def create_binary_mask(ground_truth):
    myo_val = 2
    mask = np.zeros_like(ground_truth)
    mask[ground_truth == myo_val] = 1

    return mask

def add_to_dict_info(all_patients_info : dict, patient, abspath, files):
    cfg = "Info.cfg"

    if cfg not in files:
        raise
    
    info_path = os.path.join(abspath, cfg)
    with open(info_path, 'r') as f:
        lines = f.readlines()

    pat_dict = {}
    for line in lines:
        key, value = line.strip('\n').split(': ')
        pat_dict[key] = value

    all_patients_info.append(pat_dict)

def save_mask(patient, abspath, files):

    for filename in files:
        if "_gt.nii" not in filename: continue

        path_gt = os.path.join(abspath, filename)
        gt = nib.load(path_gt).get_fdata()
        mask = create_binary_mask(gt)

        save_path = path_gt.split('.nii.gz')[0] + '.npy'
        np.save(save_path, mask, allow_pickle=True)


# dirpath = "data\\NORpatients"
dirpath = os.path.join('data', 'NORpatients')
pat_infos = []
all_pats = list(os.walk(dirpath))
print(f'walk: {all_pats}')

for patient, (abspath, folders, files) in zip(all_pats[0][1], all_pats[1:]):

    add_to_dict_info(pat_infos, patient, abspath, files)
    if pat_infos[-1]['Group'] == 'NOR':
        save_mask(patient, abspath, files)
    pass

with open(os.path.join(dirpath, 'pat_info.pkl'), 'wb') as f:
    pickle.dump(pat_infos, f)



nors = [dic['Group'] for dic in pat_infos]
if len(set(nors)) == 1 and list(set(nors))[0] == 'NOR':
    print("All patients are normal")


pass