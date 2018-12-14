import numpy as np 
from data import getReadyData, getRawData
from gen_plots import lundPlot
import matplotlib.pyplot as plt

def mass_get():
    X_train, _, _, _, _, _ = getReadyData()
    tp = np.load('true_pos.npy')
    fp = np.load('false_pos.npy')
    fn = np.load('false_neg.npy')
    tn = np.load('true_neg.npy')

    tp_mass = X_train[tp, 0, 3]
    fp_mass = X_train[fp, 0, 3]
    fn_mass = X_train[fn, 0, 3]
    tn_mass = X_train[tn, 0, 3]

    

def model_to_val():
    X_train, _, _, y_train, _, _ = getReadyData()
    from keras.models import load_model
    model = load_model('./hand_made_models/128_2_GRU_0.5_sigmoid_adam_0.03_0.09.hdf5')
    y_hat = model.predict(X_train, verbose=1)

    y_hat = np.argmax(y_hat, axis=1)
    y_train = np.argmax(y_train, axis=1)

    np.save('true_pos.npy', np.logical_and(y_hat == 1, y_train == 1))
    np.save('false_pos.npy', np.logical_and(y_hat == 1, y_train == 0))
    np.save('false_neg.npy', np.logical_and(y_hat == 0, y_train == 1))
    np.save('true_neg.npy', np.logical_and(y_hat == 0, y_train == 0))

def lund_diffs(isolate = 0):
    X_train, _, _, _, _, _ = getReadyData()
    tp = np.load('true_pos.npy')
    fp = np.load('false_pos.npy')
    fn = np.load('false_neg.npy')
    tn = np.load('true_neg.npy')

    if (isolate != 0): # advance sort
        lundPlot(X_train[tp, isolate-1:isolate], 'True Positives (W c.a. W)')
        lundPlot(X_train[fp, isolate-1:isolate], 'False Positives (QCD c.a. W)')
        lundPlot(X_train[fn, isolate-1:isolate], 'False Negatives (W c.a. QCD)')
        lundPlot(X_train[tn, isolate-1:isolate], 'True Negatives (QCD c.a. QCD)')
    else:
        lundPlot(X_train[tp], 'True Positives (W c.a. W)')
        lundPlot(X_train[fp], 'False Positives (QCD c.a. W)')
        lundPlot(X_train[fn], 'False Negatives (W c.a. QCD)')
        lundPlot(X_train[tn], 'True Negatives (QCD c.a. QCD)')

    plt.show()

def main():
    lund_diffs()

if __name__ == '__main__':
  main()