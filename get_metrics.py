from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
from partition_train_test import get_partitioned_data, get_test_data
import metrics

batch_size = 1

with open('imgs_cropped.pkl', 'rb') as f:
    df = pickle.load(f)

_, gen_val = get_partitioned_data(df, batch_size)
df = None

gen_test = get_test_data('imgs_test_cropped.pkl')

model = keras.models.load_model("model_epochs30_batchsz16_noPatMix.keras")

preds = {}
trues = {}

preds['val'] = model.predict(gen_val)
preds['test'] = model.predict(gen_test)

trues['val'] = [gen_val[i][1][0] for i in range(len(gen_val))]
trues['test'] = [gen_test[i][1][0] for i in range(len(gen_test))]

metrics.summary(trues['val'], preds['val'], cohort="val")
_, threshold = metrics.dice_threshold(trues['val'], preds['val'])
metrics.summary(trues['test'], preds['test'], cohort="test", threshold=threshold)

pass

