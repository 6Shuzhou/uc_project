import numpy as np

x = np.load(r"logs/medians/test/1/labels_sub01.npy")

for i in range(len(x)):
    print(x[i])
    