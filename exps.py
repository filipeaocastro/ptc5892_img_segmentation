import pickle
import numpy as np
from tensorflow import keras
from partition_train_test import get_partitioned_data
from classification_test import get_model

batch_size = 32
with open('imgs_cropped.pkl', 'rb') as f:
    df = pickle.load(f)

unique_vals_im = np.unique([img.shape for img in df['image']])
unique_vals_mk = np.unique([img.shape for img in df['mask']])

print(f"img shape: + {df['image'][5].shape}")
print(f"mask shape: + {df['mask'][5].shape}")

print(f"img unique: {unique_vals_im}")
print(f"mask unique: {unique_vals_mk}")