import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

current_palette = sns.color_palette()

labels = ['T1', 'T2', 'T3', 'S1', 'S2', 'S3']
option = 1
err_tf = []
loff_tr = []
if option is 1:
    err_tf = [0.9351351351, 0.950166113, 0.9441176471, 0.6332931242, 0.3582089552, 0.6]
    loss_tr = [0.9496046541, 0.9375124775, 0.9447177843, 0.6125557536, 0.3586338299, 0.5717791655]
else:
    err_tf = [0.972361343, 0.9868766404, 0.9756957328, 0.6812941614, 0.676782722, 0.6782802076]
    loss_tr = [0.9486309656, 0.9735684302, 0.9497414305, 0.5032836847, 0.5004375754, 0.4855653999]
    

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(figsize=(20, 6))
rects1 = ax.bar(x - width/2 - 0.01, err_tf, width, label='Err transferability', color=current_palette[0], zorder=3, alpha = 0.93)
rects2 = ax.bar(x + width/2 + 0.01, loss_tr, width, label='Loss transferability', color=current_palette[3], zorder=3, alpha = 0.93)

ax.grid(zorder=0, color="white", linewidth=1)
ax.set_facecolor('#e7e7f0')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Transferability', fontsize=40)
ax.set_xlabel('Source model', fontsize=40)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.legend()

plt.ylim(0, 1.1) 
plt.yticks(np.arange(0, 1.1, step=0.2))


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
ax.legend(fontsize=30)

fig.tight_layout()
if option is 1:
    plt.savefig('data-pic/mnist-trans.pdf')
else:
    plt.savefig('data-pic/cifar-trans.pdf')

plt.show()