import h5py
import numpy as np

D_PATH = '../Data/'
W = 'W'
Z = 'Z'
JZ4W = 'jz4w'
JZ3W = 'jz3w'
JZ7W = 'jz7w'

SIG = W
BG = JZ3W

RAW_SIG_PATH = D_PATH + SIG + '.h5'
RAW_BG_PATH = D_PATH + BG + '.h5'

SIG_PATH = D_PATH + SIG + 'npy'
BG_PATH = D_PATH + BG + 'npy'

def getRawData():
    fSig = h5py.File(RAW_SIG_PATH, 'r')
    dSig = np.array(fSig['lundjets_InDetTrackParticles'])
    fSig.close()

    fBg = h5py.File(RAW_BG_PATH, 'r')
    dBg = np.array(fBg['lundjets_InDetTrackParticles'])
    fBg.close()

    dSig = dSig.reshape(2*dSig.shape[0], 20, 5)
    dBg = dBg.reshape(2*dBg.shape[0], 20, 5)

    return [dSig, dBg]

def cutData(data, ptMin, ptMax, isSig=1, pt2Min=0, pt2Max=1000):
  return np.squeeze(data[np.where(np.logical_and(data[:, 0, 2] < ptMax, data[:, 0, 2] > ptMin)), :, :])

def evenData(data1, data2):
  size1= data1.shape[0]
  size2 = data2.shape[0]
  if size1 > size2:
    data1 = data1[:size2]
  else:
    data2 = data2[:size1]

  return [data1, data2]

def saveProData(sig, bg):
  np.save(SIG_PATH, sig)
  np.save(BG_PATH, bg)
  

if __name__ == '__main__':
  getRawData()