import matplotlib.pyplot as plt
import numpy as np

# xs_interp=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_spacio_success_3/xs_interp.npy")
# xs=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_spacio_success_3/xs.npy")
# us=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_spacio_success_3/us.npy")

xs_interp=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_temporal_success_1/xs_interp.npy")
xs=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_temporal_success_1/xs.npy")
us=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_temporal_success_1/us.npy")


fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(20, 10))

for i in range(7):
    axs[i, 0].stem(xs[:, 14], xs[:, i], markerfmt='ro', linefmt='r-', basefmt='k:')
    axs[i,0].set_ylabel('q{}'.format(i+1))
    axs[i, 1].stem(xs[:, 15], xs[:, i+7], markerfmt='bo', linefmt='b-', basefmt='k:')
    # axs[i, 1].set_xlabel('s2')
    # axs[i, 1].set_ylabel('q{}'.format(i + 1))

axs[-1,0].set_xlabel('s1')
axs[-1,1].set_xlabel('s2')

fig.tight_layout()
plt.savefig('/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_temporal_success_1/q_vs_s.png')
plt.show()