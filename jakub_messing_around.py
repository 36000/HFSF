from data import getRawData
from matplotlib import pyplot as plt

sig, bg = getRawData()
print(sig.shape)
print(bg.shape)

sig = sig[:, 0, 3]
bg = bg[:, 0, 3]

sig = sig[sig > 40]
bg = bg[bg > 40]

sig = sig[sig < 300]
bg = bg[bg < 300]

plt.hist(sig, density=True, label='W', histtype='step', bins=400)
plt.hist(bg, density=True, label='QCD', histtype='step', bins=400)

plt.legend()
plt.show()
