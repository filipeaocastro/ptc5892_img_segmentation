import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


from tools.utils import imgRetriever

def get_partitioned_data(df, batch_size, random_seed=1, val_pats=4):

    # df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    patients = np.unique(df['patient_id'])
    print(patients)
    np.random.seed(random_seed)
    patients = np.random.choice(patients, size=val_pats, replace=False)
    pat_mask = df['patient_id'].isin(patients)

    print(patients)
    
    val_df = df[pat_mask].sample(frac=1.0, random_state=random_seed)
    train_df = df.drop(val_df.index, axis=0).sample(frac=1.0, random_state=random_seed)

    val_df = val_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    img_size = df['image'][0].shape

    train_gen = imgRetriever(batch_size, img_size, train_df)
    val_gen = imgRetriever(batch_size, img_size, val_df)

    return train_gen, val_gen

def get_test_data(dirpath, batch_size):

    path = os.path.join(dirpath, 'imgs_test_cropped.pkl')

    with open(path, 'rb') as f:
        df = pickle.load(f)
    
    img_size = df['image'][0].shape
    test_gen = imgRetriever(batch_size, img_size, df)

    return test_gen

if __name__ == '__main__':

    batch_size = 32
    with open('imgs_cropped.pkl', 'rb') as f:
        df = pickle.load(f)

    plt.imshow(df['mask'][5], cmap='gray')
    plt.show()

    train, val = get_partitioned_data(df, batch_size)
    df = None
    pass
