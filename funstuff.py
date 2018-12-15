import numpy as np
from data import getReadyData
from gen_plots import lundPlot
import matplotlib.pyplot as plt


def mass_get():
    x_train, _, _, _, _, _ = getReadyData(include_mass=True)

    tp = np.load('true_pos.npy')
    fp = np.load('false_pos.npy')
    fn = np.load('false_neg.npy')
    tn = np.load('true_neg.npy')  # How to generate those?

    tp_mass = x_train[tp, 0, 3]
    fp_mass = x_train[fp, 0, 3]
    fn_mass = x_train[fn, 0, 3]
    tn_mass = x_train[tn, 0, 3]

    bins = np.linspace(0, 250, 500)

    fig_1 = plt.subplot(2, 1, 1)
    fig_1.hist(tp_mass, label='Mass W c.a. W', color='orange', histtype='step', bins=bins, density=True)
    fig_1.hist(fp_mass, label='Mass QCD c.a. W', color='blue', histtype='step', bins=bins, density=True)
    fig_1.legend()
    fig_1.set_title('Ungroomed jet mass distribution')

    fig_2 = plt.subplot(2, 1, 2)
    fig_2.hist(tn_mass, label='Mass QCD c.a. QCD', color='blue', histtype='step', bins=bins, density=True)
    fig_2.hist(fn_mass, label='Mass W c.a. QCD', color='orange', histtype='step', bins=bins, density=True)
    fig_2.legend()
    fig_2.set_xlabel('Ungroomed Mass (GeV)')

    plt.tight_layout()
    plt.savefig('Pictures/mass_histogram')
    plt.show()


def model_to_val():
    x_train, _, _, y_train, _, _ = getReadyData()
    from keras.models import load_model
    model = load_model('./hand_made_models/64_5_GRU_0.5_sigmoid_adam_0.03_0.09.hdf5')
    y_hat = model.predict(x_train, verbose=1)

    y_hat = np.argmax(y_hat, axis=1)
    y_train = np.argmax(y_train, axis=1)

    np.save('true_pos.npy', np.logical_and(y_hat == 1, y_train == 1))
    np.save('false_pos.npy', np.logical_and(y_hat == 1, y_train == 0))
    np.save('false_neg.npy', np.logical_and(y_hat == 0, y_train == 1))
    np.save('true_neg.npy', np.logical_and(y_hat == 0, y_train == 0))


def lund_diffs(isolate=21):
    x_train, _, _, _, _, _ = getReadyData()
    tp = np.load('true_pos.npy')
    fp = np.load('false_pos.npy')
    fn = np.load('false_neg.npy')
    tn = np.load('true_neg.npy')

    if isolate != 0:  # advance sort
        lundPlot(x_train[tp, isolate - 1:isolate], 'True Positives (W c.a. W)')
        lundPlot(x_train[fp, isolate - 1:isolate], 'False Positives (QCD c.a. W)')
        lundPlot(x_train[fn, isolate - 1:isolate], 'False Negatives (W c.a. QCD)')
        lundPlot(x_train[tn, isolate - 1:isolate], 'True Negatives (QCD c.a. QCD)')
    else:
        lundPlot(x_train[tp], 'True Positives (W c.a. W)')
        lundPlot(x_train[fp], 'False Positives (QCD c.a. W)')
        lundPlot(x_train[fn], 'False Negatives (W c.a. QCD)')
        lundPlot(x_train[tn], 'True Negatives (QCD c.a. QCD)')

    plt.show()


def main():
    # model_to_val()
    mass_get()
    # lund_diffs(1)


if __name__ == '__main__':
    main()
