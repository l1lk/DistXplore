import numpy as np
import matplotlib.pyplot as plt

a = np.load("./vae_svhn/seeds/25_9_4_orig.npy")
b = np.load("./vae_svhn/25_9_4.npy")

plt.imshow(a[0])
plt.savefig("./vae_cifar_test3/test1.png")
plt.imshow(b[0])
plt.savefig("./vae_cifar_test3/test2.png")
