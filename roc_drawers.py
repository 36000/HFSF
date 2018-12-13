import numpy as np


# ROC curve drawer, given generator.
def __draw_roc(model, x, y, verbose=2, save_tag=''):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    y_predict = model.predict(x, verbose=verbose)

    tpr, fpr, auc = __multi_roc_data(y_true=y, y_pred=y_predict)

    # Needed to create two subplots with different sizes.
    # If other ratios are needed change height_ratios.
    plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    main = plt.subplot(gs[0])
    main.set_yscale('log')
    main.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlim([0.1, 1.])
    plt.ylim([1., 10.**3])

    ratio = plt.subplot(gs[1])
    ratio.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlim([0.1, 1.])
    plt.ylim([0.1, 1.1])

    main.plot(np.arange(0.1, 1.0, 0.001), np.divide(1., np.arange(0.1, 1.0, 0.001)), 'k--', label='Luck (AUC = 0.5000)')

    print('Creating curve...')
    main.plot(tpr, fpr, color='b',
              label='LSTM (AUC = %0.4f)'.format(auc))

    main.set_ylabel("1 / [Background Efficiency]")
    main.set_title("ROC Curve for LSTM")
    main.legend(loc=1, frameon=False)
    plt.tight_layout()
    plt.savefig("ROC Curve {}".format(save_tag))
    plt.clf()
    print('ROC Curve successfully created.')


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
