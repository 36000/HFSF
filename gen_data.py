import numpy as np
import matplotlib.pyplot as plt

from data import getRawData, cutData, evenData, saveProData, SIG, BG
from gen_plots import lundPlot

def main():
  sig, bg = getRawData()

  '''
  newWindow()
  plt.hist(bg[:, 0, 0, 2])

  newWindow()
  plt.hist(bg[:, 0, 0, 2], range=(200, 600))
  '''

  sig = cutData(sig, 200, 400)
  bg = cutData(bg, 200, 400)

  sig, bg = evenData(sig, bg)

  lundPlot(sig, SIG)
  lundPlot(bg, BG)
  plt.show()

  saveProData(sig, bg)
  
  '''
  pt2_f = sig[:, :, :, 4].flatten()
  pt2_f = pt2_f[pt2_f != 10]
  newWindow()
  plt.hist(pt2_f)

  newWindow()
  binwidth = 1
  plt.hist(pt2_f, bins=range(int(np.min(pt2_f)), int(np.max(pt2_f)) + binwidth, binwidth), range=(0, 20))
  '''
  print('Data Shape: ' + str(sig.shape))



'''
  for i in range(sig.shape[0]):
    deltaR = sig[i, 0, :, 0]
    Z = sig[i, 0, :, 1]

    deltaR = deltaR[deltaR != 10]
    Z = Z[Z != 10]

    plt.plot(np.log(np.divide(1, deltaR)), np.log(np.divide(1, Z)))
    plt.show()
    plt.clf()
'''



if __name__ == '__main__':
  main()