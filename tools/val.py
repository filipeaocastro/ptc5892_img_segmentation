
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
from partition_train_test import get_partitioned_data, get_test_data
import metrics

def threshold_binary_images(pred_prob_maps, threshold=0.5):
    # Apply threshold to the probability maps
    binary_images = (pred_prob_maps > threshold).astype(np.uint8)
    return binary_images


test_data = False

if test_data:
    gen = get_test_data('imgs_test_cropped.pkl')
else:
    batch_size = 1
    with open('imgs_cropped.pkl', 'rb') as f:
        df = pickle.load(f)

    _, gen = get_partitioned_data(df, batch_size)
    df = None

model = keras.models.load_model("model_epochs25_batchsz16_noPatMix.keras")

preds = model.predict(gen)

trues = [gen[i][1][0] for i in range(len(gen))]

# for i in range(10):
#     f, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     ax1.imshow(val_gen.__getitem__(0)[0][i], cmap='gray')
#     ax2.imshow(val_gen.__getitem__(0)[1][i], cmap='gray')
#     ax3.imshow(bin_preds[i], cmap='gray')
#     plt.show(block=True)

# metrics.summary(trues, preds, cohort="val")
# _, threshold = metrics.dice_threshold(trues, preds)
# metrics.summary(trues, preds, cohort="test", threshold=threshold)

_, th = metrics.dice_threshold(trues, preds, plot=False)
bin_preds = threshold_binary_images(preds, th)

samples = 5
for i in range(0, len(bin_preds), samples):
    f, axs = plt.subplots(3, samples)
    for j in range(samples):
        axs[0, j].imshow(gen[i+j][0][0], cmap='gray')
        axs[1, j].imshow(gen[i+j][1][0], cmap='gray')
        axs[2, j].imshow(bin_preds[j+i], cmap='gray')
    plt.show()

pass

# dice = (2*np.sum(true[pred==1])+0.001)/(np.sum(true)+np.sum(pred)+0.001)
# media e std de todos