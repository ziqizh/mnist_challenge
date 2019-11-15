import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

current_palette = sns.color_palette()


labels = ['T1', 'T2', 'T3', 'S1', 'S2', 'S3']
err_tf = [0.972361343, 0.9868766404, 0.9756957328, 0.6812941614, 0.676782722, 0.6782802076]
loss_tr = [0.9486309656, 0.9735684302, 0.9497414305, 0.5032836847, 0.5004375754, 0.4855653999]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, err_tf, width, label='Err transferability', color=current_palette[0], zorder=3, alpha = 0.93)
rects2 = ax.bar(x + width/2, loss_tr, width, label='Loss transferability', color=current_palette[3], zorder=3, alpha = 0.93)

ax.grid(zorder=0, color="white", linewidth=1)
ax.set_facecolor('#e7e7f0')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Transferability', fontsize=12)
ax.set_xlabel('Source model', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.legend()

plt.ylim(0, 1.1) 
plt.yticks(np.arange(0, 1.1, step=0.2))


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('pic/cifar-trans.png')

plt.show()