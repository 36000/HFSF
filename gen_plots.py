import numpy as np
import matplotlib.pyplot as plt

from random import randint
import sys

from data import getProData, SIG, BG

def newWindow(title='UNTITLED'):
  plt.figure(randint(-sys.maxsize, sys.maxsize)).suptitle(title, fontsize=16)

def lundifyPlt(plt):
  plt.xlabel('ln 1/deltaR')
  plt.ylabel('ln 1/Z')
  plt.colorbar()

def lundPlot(data, dataName='?', binMin=0, binMax=8, binWidth=1, zoom=50):
  deltaR = data[:, :, 0].flatten()
  Z = data[:, :, 1].flatten()

  deltaR = deltaR[deltaR != 10]
  Z = Z[Z != 10]

  logiDeltaR = np.log(np.divide(1, deltaR))
  logiZ = np.log(np.divide(1, Z))

  newWindow('Lund Plot for ' + dataName)
  plt.hist2d(logiDeltaR, logiZ, bins=np.linspace(binMin, binMax, (binMax-binMin)*zoom), cmap='Spectral')
  lundifyPlt(plt)

def lundDiff(sig, bg, binMin=0, binMax=8, binWidth=1, zoom=50):
  deltaR = sig[:, :, 0].flatten()
  Z = sig[:, :, 1].flatten()

  deltaR = deltaR[deltaR != 10]
  Z = Z[Z != 10]

  logiDeltaR = np.log(np.divide(1, deltaR))
  logiZ = np.log(np.divide(1, Z))

  data1hist, _, _ = np.histogram2d(logiDeltaR, logiZ, bins=np.linspace(binMin, binMax, (binMax-binMin)*zoom))

  deltaR = bg[:, :, 0].flatten()
  Z = bg[:, :, 1].flatten()

  deltaR = deltaR[deltaR != 10]
  Z = Z[Z != 10]

  logiDeltaR = np.log(np.divide(1, deltaR))
  logiZ = np.log(np.divide(1, Z))

  data2hist, _, _ = np.histogram2d(logiDeltaR, logiZ, bins=np.linspace(binMin, binMax, (binMax-binMin)*zoom))

  diff_hist = np.rot90(np.subtract(data1hist, data2hist))

  newWindow(SIG + ' minus ' + BG + ' Plot')
  plt.imshow(diff_hist, cmap='Spectral')

def main():
    sig, bg = getProData()
    lundPlot(sig, SIG)
    lundPlot(bg, BG)
    lundDiff(sig, bg)
    plt.show()

if __name__ == '__main__':
  main()