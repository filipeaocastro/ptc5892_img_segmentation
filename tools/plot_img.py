import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib


def create_binary_mask(ground_truth):
    myo_val = 2
    mask = np.zeros_like(ground_truth)
    mask[ground_truth == myo_val] = 1

    return mask

if __name__ == '__main__':

    dirname = "data\\NORpatients\\patient061"
    filename_img = "patient061_frame01.nii.gz"
    filename_gt = "patient061_frame01_gt.nii.gz"
    # filename_4d = "patient001_4d.nii.gz"
    path_img = os.path.join(dirname, filename_img)
    path_gt = os.path.join(dirname, filename_gt)
    # path_4d = os.path.join(dirname, filename_4d)

    img = nib.load(path_img)
    gt = nib.load(path_gt)
    # d4 = nib.load(path_4d)
    i = 1
    plt.imshow(img.get_fdata()[:, :, i], cmap='gray')
    plt.imshow(gt.get_fdata()[:, :, i], alpha=0.4)   
    plt.show()

    exit()

    plot = True

    mask = create_binary_mask(gt.get_fdata())
    if plot:
        for i in range(10):
            plt.imshow(img.get_fdata()[:, :, i])
            plt.imshow(mask[:, :, i], alpha=0.3, cmap=plt.cm.gray)
            plt.show(block=True)

pass

