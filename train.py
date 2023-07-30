import pickle
import numpy as np
from tensorflow import keras
from partition_train_test import get_partitioned_data
from classification_test import get_model

batch_size = 32
with open('imgs_cropped.pkl', 'rb') as f:
    df = pickle.load(f)

train_gen, val_gen = get_partitioned_data(df, batch_size)

model = get_model(df['image'][0].shape, 2)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")

# callbacks = [
#     keras.callbacks.ModelCheckpoint("checkpoint_segmentation.keras", save_best_only=True)
# ]

# Train the model, doing validation at the end of each epoch.
epochs = 8
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
model.fit(train_gen, epochs=epochs, validation_data=val_gen)
model.save("model.keras")