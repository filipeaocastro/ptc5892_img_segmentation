import numpy as np
import pandas as pd
from tensorflow import keras

class imgRetriever(keras.utils.Sequence):
    """Helper to iterate over the data"""

    def __init__(self, batch_size, img_size, imgs_df):
        self.batch_size = batch_size
        self.img_size = img_size
        self.imgs_df = imgs_df

    def __len__(self):
        return len(self.imgs_df['image']) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        # batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        # batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, img in enumerate(self.imgs_df['image'][i : i + self.batch_size]):
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, img in enumerate(self.imgs_df['mask'][i : i + self.batch_size]):
            y[j] = img

        return x, y