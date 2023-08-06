# from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from tensorflow.keras import layers

def get_model(img_size):
    keras.backend.clear_session()
    inputs = keras.Input(shape=img_size)

    # Contracting Path (Encoder)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # new bottleneck

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    # pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # # Bottleneck
    # conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    # conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # # Expanding Path (Decoder)
    # up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    # merge6 = layers.concatenate([conv4, up6], axis=3)
    # conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    # conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.BatchNormalization()(conv9)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model


if __name__ == '__main__':
    # with open('imgs_cropped.pkl', 'rb') as f:
    #     df = pickle.load(f)

    random_seed = 1
    # df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # img_size = df['image'][0].shape
    img_size = (128, 128, 1)
    num_classes = 2
    batch_size = 16

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_model(img_size)
    model.summary()