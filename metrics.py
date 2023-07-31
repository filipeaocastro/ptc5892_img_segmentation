import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error


def threshold_binary_images(pred_prob_maps, threshold=0.5):
    # Apply threshold to the probability maps
    binary_images = (pred_prob_maps > threshold).astype(np.uint8)
    return binary_images

def _mean(trues, preds, metric):
    metrics = [metric(true, pred) for true, pred in zip(trues, preds)]
    return np.mean(metrics), np.std(metrics)

def dice(true, pred, threshold=0.2):
    pred = threshold_binary_images(pred, threshold)
    return (2 * np.sum(true[pred == 1]) + 0.001) / (np.sum(true) + np.sum(pred) + 0.001)

def dice_mean(trues, preds, threshold=0.2):
    dices = [dice(true, pred, threshold) for true, pred in zip(trues, preds)]
    return np.mean(dices), np.std(dices), threshold

def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def ssim(true, pred):
    return sk_ssim(true, pred, channel_axis=-1, data_range=1)

def ssim_mean(trues, preds):
    return _mean(trues, preds, ssim)

def nrmse(true, pred, method='intensity'):
    if method == 'range':
        return rmse(true, pred) / (true.max() - true.min())
    elif method == 'mean':
        return rmse(true, pred) / np.mean(true)
    elif method == 'intensity':
        return np.sqrt(np.power(true - pred, 2) / (np.power(true, 2) + 0.001))
    
def nrmse_mean(trues, preds, method='intensity'):
    nrmses = [nrmse(true, pred, method) for true, pred in zip(trues, preds)]
    return np.mean(nrmses), np.std(nrmses)

    
def dice_threshold(trues, preds, plot=False):

    thresholds = np.linspace(0.05, 0.95, 19)
    dices = []

    for t in thresholds:
        bin_preds = threshold_binary_images(preds, threshold=t)
        mean_dice = [dice(true, pred) for true, pred in zip(trues, bin_preds)]
        dices.append(np.mean(mean_dice))
    
    if plot:
        plt.plot(thresholds, dices)
        plt.xlabel('Threshold')
        plt.ylabel("Dice")
        plt.show()

    return max(dices), thresholds[np.argmax(dices)] 

def summary(trues, preds, cohort="val", threshold=0.2):

    if cohort == 'val':
        print(f"\n##### Metrics Summary for validation #####")
        print("NRMSE: %.3f ± %.3f" % nrmse_mean(trues, preds))
        print("SSIM: %.3f ± %.3f" % ssim_mean(trues, preds))
        print("Best Dice: %.3f - for threshold %.3f" % dice_threshold(trues, preds))

    else:
        print(f"\n##### Metrics Summary for testing #####")
        print("NRMSE: %.3f ± %.3f" % nrmse_mean(trues, preds))
        print("SSIM: %.3f ± %.3f" % ssim_mean(trues, preds))
        print("Dice: %.3f ± %.3f (threshold = %.2f)" % dice_mean(trues, preds, threshold))
        # print("Best Dice: %.3f - for threshold %.3f" % dice_threshold(trues, preds))