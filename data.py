import h5py
import numpy as np
from sklearn.model_selection import train_test_split

D_PATH = '../data/'
W = 'W'
Z = 'Z'
JZ4W = 'jz4w'
JZ3W = 'jz3w'
JZ7W = 'jz7w'

SIG = W
BG = JZ3W

RAW_SIG_PATH = D_PATH + SIG + '.h5'
RAW_BG_PATH = D_PATH + BG + '.h5'

SIG_PATH = D_PATH + SIG + '.npy'
BG_PATH = D_PATH + BG + '.npy'

READY_DATA = D_PATH + 'ready_data.h5'


def getRawData():
    fSig = h5py.File(RAW_SIG_PATH, 'r')
    dSig = np.array(fSig['lundjets_InDetTrackParticles'])
    fSig.close()

    fBg = h5py.File(RAW_BG_PATH, 'r')
    dBg = np.array(fBg['lundjets_InDetTrackParticles'])
    fBg.close()

    dSig = dSig.reshape(2 * dSig.shape[0], 20, 5)
    dBg = dBg.reshape(2 * dBg.shape[0], 20, 5)

    return [dSig, dBg]


def cutData(data, ptMin, ptMax, isSig=1, pt2Min=0, pt2Max=1000):
    return np.squeeze(data[np.where(np.logical_and(data[:, 0, 2] < ptMax, data[:, 0, 2] > ptMin)), :, :])


def evenData(data1, data2):
    size1 = data1.shape[0]
    size2 = data2.shape[0]
    if size1 > size2:
        data1 = data1[:size2]
    else:
        data2 = data2[:size1]

    return [data1, data2]


def saveProData(sig, bg):
    np.save(SIG_PATH, sig)
    np.save(BG_PATH, bg)


def getProData():
    return [np.load(SIG_PATH), np.load(BG_PATH)]


def prepData():
    from keras.utils.np_utils import to_categorical
    sig = np.load(SIG_PATH)[:, :, 0:2]
    bg = np.load(BG_PATH)[:, :, 0:2]

    y = np.concatenate((np.ones(sig.shape[0]), np.zeros(bg.shape[0])))
    X = np.concatenate((sig, bg), axis=0)

    y = to_categorical(y, 2)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    with h5py.File(READY_DATA, 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('X_test', data=X_test)
        hf.create_dataset('y_test', data=y_test)
        hf.create_dataset('X_val', data=X_val)
        hf.create_dataset('y_val', data=y_val)


def getReadyData():
    with h5py.File(READY_DATA, 'r') as hf:
        X_train = np.array(hf['X_train'])
        y_train = np.array(hf['y_train'])
        X_test = np.array(hf['X_test'])
        y_test = np.array(hf['y_test'])
        X_val = np.array(hf['X_val'])
        y_val = np.array(hf['y_val'])
    return [X_train, X_val, X_test, y_train, y_val, y_test]


if __name__ == '__main__':
    prepData()
