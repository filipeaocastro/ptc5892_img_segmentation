
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
from partition_train_test import get_partitioned_data

model = keras.models.load_model("model.keras")

batch_size = 32
with open('imgs_cropped.pkl', 'rb') as f:
    df = pickle.load(f)

train_gen, val_gen = get_partitioned_data(df, batch_size)

val_preds = model.predict(val_gen)

for i in range(10):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(val_gen.__getitem__(0)[0][i], cmap='gray')
    ax2.imshow(val_gen.__getitem__(0)[1][i], cmap='gray')
    ax3.imshow(val_preds[i], cmap='gray')
    plt.show(block=True)

pass