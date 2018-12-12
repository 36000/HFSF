import h5py
import numpy as np

D_PATH = '../Data/'
W = 'W.h5'
Z = 'Z.h5'
JZ4W = 'jz4w.h5'
JZ7W = 'jz7w.h5'
CUT = 'cut_'

SIG = W
BG = Z

SIG_PATH = D_PATH + SIG
BG_PATH = D_PATH + BG

def getData():
    fSig = h5py.File(SIG_PATH, 'r')
    dSig = np.array(fSig['lundjets_InDetTrackParticles'])
    fSig.close()

    fBg = h5py.File(BG_PATH, 'r')
    dBg = np.array(fBg['lundjets_InDetTrackParticles'])
    fBg.close()

    return [dSig, dBg]

def cutData(data, ptMin, ptMax, isSig=1):
  cData = data[np.where(np.logical_and(data[:, 0, 0, 2] < ptMax, data[:, 0, 0, 2] > ptMin)), :, :, :]
  if isSig:
    np.save(D_PATH + CUT + SIG, cData)
  else:
    np.save(D_PATH + CUT + BG, cData)
  return cData
  

if __name__ == '__main__':
  getData()