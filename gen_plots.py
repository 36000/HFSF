from data import getData, cutData
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
  sig, bg = getData()

  plt.hist(sig[:, 0, 0, 2])
  plt.show()
  plt.clf()

  plt.hist(sig[:, 0, 0, 2], range=(200, 600))
  plt.show()
  plt.clf()

  sig = cutData(sig, 200, 450)

  print('Data Shape: ' + str(sig.shape))

  deltaR = sig[0, 0, :, 0]
  Z = sig[0, 0, :, 1]

  plt.plot(np.log(np.divide(1, deltaR)), np.log(np.divide(1, Z)))
  plt.show()
  plt.clf()



if __name__ == '__main__':
  main()