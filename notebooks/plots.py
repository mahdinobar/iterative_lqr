import matplotlib.pyplot as plt
import numpy as np

def plot_q_vs_s(file_dir):
    xs=np.load("{}/xs.npy".format(file_dir))
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
    plt.savefig('{}/q_vs_s.png'.format(file_dir))
    plt.show()

def plot_ds_bar(file_dir):
    us=np.load("{}/us.npy".format(file_dir))
    # fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(20, 10))
    # for i in range(7):
    #     axs[i, 0].stem(xs[:, 14], xs[:, i], markerfmt='ro', linefmt='r-', basefmt='k:')
    #     axs[i,0].set_ylabel('q{}'.format(i+1))
    #     axs[i, 1].stem(xs[:, 15], xs[:, i+7], markerfmt='bo', linefmt='b-', basefmt='k:')
    #     # axs[i, 1].set_xlabel('s2')
    #     # axs[i, 1].set_ylabel('q{}'.format(i + 1))
    # axs[-1,0].set_xlabel('s1')
    # axs[-1,1].set_xlabel('s2')
    # fig.tight_layout()
    # plt.savefig('{}/q_vs_s.png'.format(file_dir))
    # plt.show()

    labels = np.linspace(0,us.shape[0]-1,us.shape[0], dtype='int').astype('str')
    us1 = us[:, 14]
    us2 = us[:, 15]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x - width / 2, us1, width, label='robot arm 1', color='r')
    ax.bar(x + width / 2, us2, width, label='robot arm 2', color='b')
    ax.legend()
    ax.set_ylabel('phase')
    ax.set_xlabel('step')
    fig.tight_layout()
    plt.show()
    plt.savefig('{}/ds_bar.png'.format(file_dir))

if __name__ == "__main__":
    file_dir='/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/tailor_temporal_success_1'
    # plot_q_vs_s(file_dir)
    plot_ds_bar(file_dir)