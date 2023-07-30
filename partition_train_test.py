import random
import pickle

from utils import imgRetriever

def get_partitioned_data(df, batch_size, random_seed=1):

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = df.sample(frac=0.2, random_state=random_seed)
    train_df = df.drop(val_df.index, axis=0)

    img_size = df['image'][0].shape
    batch_size = 32

    train_gen = imgRetriever(batch_size, img_size, train_df)
    val_gen = imgRetriever(batch_size, img_size, val_df)

    return train_gen, val_gen

if __name__ == '__main__':

    batch_size = 32
    with open('imgs_cropped.pkl', 'rb') as f:
        df = pickle.load(f)

    train, val = get_partitioned_data(df, batch_size)


