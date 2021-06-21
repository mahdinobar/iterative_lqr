import matplotlib.pyplot as plt
import numpy as np

xs_interp=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/xs_interp.npy")

x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))

plt.stem(x, y)
plt.show()