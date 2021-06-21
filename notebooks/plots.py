import matplotlib.pyplot as plt
import numpy as np

xs_interp=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_spacio_success_3/xs_interp.npy")
xs=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_spacio_success_3/xs.npy")
us=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_spacio_success_3/us.npy")

x = xs[:,15]
y = xs[:,7]

plt.stem(x, y)
plt.show()