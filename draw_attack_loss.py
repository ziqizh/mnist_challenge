import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.switch_backend('agg')

log1 = open('data-log/measure/atta-mnist-loss-1.log')
log2 = open('data-log/measure/atta-mnist-loss-2.log')
log3 = open('data-log/measure/atta-mnist-loss-3.log')
log4 = open('data-log/measure/atta-mnist-loss-4.log')
log5 = open('data-log/measure/atta-mnist-loss-5.log')
log6 = open('data-log/measure/atta-mnist-loss-6.log')
log7 = open('data-log/measure/atta-mnist-loss-7.log')
log8 = open('data-log/measure/atta-mnist-loss-8.log')
log9 = open('data-log/measure/atta-mnist-loss-9.log')
log10 = open('data-log/measure/atta-mnist-loss-10.log')

label1 = "cpt-1"
label2 = "cpt-2"
label3 = "cpt-3"
label4 = "cpt-4"
label5 = "cpt-5"
label6 = "cpt-6"
label7 = "cpt-7"
label8 = "cpt-8"
label9 = "cpt-9"
label10 = "cpt-10"

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []
data10 = []
length = 50

log_lines1 = log1.readlines()
log_lines2 = log2.readlines()
log_lines3 = log3.readlines()
log_lines4 = log4.readlines()
log_lines5 = log5.readlines()
log_lines6 = log6.readlines()
log_lines7 = log7.readlines()
log_lines8 = log8.readlines()
log_lines9 = log9.readlines()
log_lines10 = log10.readlines()
for i in range(length):
  data1.append([eval(j) for j in log_lines1[i].split(' ')])
  data2.append([eval(j) for j in log_lines2[i].split(' ')])
  data3.append([eval(j) for j in log_lines3[i].split(' ')])
  data4.append([eval(j) for j in log_lines4[i].split(' ')])
  data5.append([eval(j) for j in log_lines5[i].split(' ')])
  data6.append([eval(j) for j in log_lines6[i].split(' ')])
  data7.append([eval(j) for j in log_lines7[i].split(' ')])
  data8.append([eval(j) for j in log_lines8[i].split(' ')])
  data9.append([eval(j) for j in log_lines9[i].split(' ')])
  data10.append([eval(j) for j in log_lines10[i].split(' ')])


print(len(data1))

x = np.array([i[0] for i in data1]) + 1

adv_loss1 = np.array([i[1] for i in data1])
adv_loss2 = np.array([i[1] for i in data2])
adv_loss3 = np.array([i[1] for i in data3])
adv_loss4 = np.array([i[1] for i in data4])
adv_loss5 = np.array([i[1] for i in data5])
adv_loss6 = np.array([i[1] for i in data6])
adv_loss7 = np.array([i[1] for i in data7])
adv_loss8 = np.array([i[1] for i in data8])
adv_loss9 = np.array([i[1] for i in data9])
adv_loss10 = np.array([i[1] for i in data10])

current_palette = sns.color_palette()

plt.plot(x, adv_loss1, color=current_palette[0], label=label1, lw=2)
plt.plot(x, adv_loss2, color=current_palette[1], label=label2, lw=2)
plt.plot(x, adv_loss3, color=current_palette[2], label=label3, lw=2)
plt.plot(x, adv_loss4, color=current_palette[3], label=label4, lw=2)
plt.plot(x, adv_loss5, color=current_palette[4], label=label5, lw=2)
plt.plot(x, adv_loss6, color=current_palette[5], label=label6, lw=2)
plt.plot(x, adv_loss7, color=current_palette[6], label=label7, lw=2)
plt.plot(x, adv_loss8, color=current_palette[7], label=label8, lw=2)
plt.plot(x, adv_loss9, color=current_palette[8], label=label9, lw=2)
plt.plot(x, adv_loss10, color=current_palette[9], label=label10, lw=2)

plt.xlabel("Attack iterations", fontsize=15)
plt.ylabel("Loss value", fontsize=15)
plt.tick_params(labelsize=10)

plt.legend(fontsize='x-large')

plt.savefig('data-pic/multiple-checkpoints-attack-loss.png')
