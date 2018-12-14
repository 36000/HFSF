import numpy as np
from keras.models import load_model
from data import getReadyData
import matplotlib.pyplot as plt


# ROC curve drawer, given generator.
def __draw_roc(model, x, y, verbose=1, label=''):
    y_predict = model.predict(x, verbose=verbose)

    tpr, fpr, auc = __multi_roc_data(y_true=y, y_pred=y_predict)

    # Needed to create two subplots with different sizes.
    # If other ratios are needed change height_ratios.
    plt.figure(figsize=(6, 8))

    main = plt.subplot()
    main.set_yscale('log')
    main.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlim([0.1, 1.])
    plt.ylim([1., 10.**3])

    main.plot(np.arange(0.1, 1.0, 0.001), np.divide(1., np.arange(0.1, 1.0, 0.001)), 'k--', label='Luck (AUC = 0.5000)')

    print('Creating curve...')
    main.plot(tpr, fpr, color='b', label='GRU (AUC = {0:.4f})'.format(auc))

    main.set_ylabel("1 / [Background Efficiency]")
    main.set_title("ROC Curve for GRU " + label)
    main.legend(loc=1, frameon=False)
    print('ROC Curve successfully created.')

    return main


# Calculates and return true positive rate, false positive rate, area under curve,
# and ratios of false positive rate with respect to false positive rate of given generator gen.
# For multi-class (categorical) model.
def __multi_roc_data(y_true, y_pred):
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    print('Creating ROC data')
    n_classes = len(y_true[0])
    gen_fpr = {}
    gen_tpr = {}
    gen_roc_auc = {}
    for i in range(n_classes):
        gen_fpr[i], gen_tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        gen_roc_auc[i] = auc(gen_fpr[i], gen_tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([gen_fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, gen_fpr[i], gen_tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr = all_fpr
    tpr = mean_tpr
    auc_score = auc(fpr, tpr)
    fpr = np.divide(1., fpr)  # Revert fpr for them weird roc curves

    return tpr, fpr, auc_score


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = getReadyData()
    model = load_model('./hand_made_models/64_5_GRU_0.5_sigmoid_adam_0.03_0.09.hdf5')
    __draw_roc(model, X_test, y_test, verbose=1, label='Test.png')
    plt.show()
    __draw_roc(model, X_train[:X_train.shape[0]//3], y_train[:X_train.shape[0]//3], verbose=1, label='Train.png')
    plt.show()
    __draw_roc(model, X_val, y_val, verbose=1, label='Val.png')
    plt.show()

if __name__ == '__main__':
  main()