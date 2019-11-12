import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.switch_backend('agg')

log1 = open('data-log/measure/atta-loss-vs-step-1.txt')
log2 = open('data-log/measure/atta-loss-vs-step-5.txt')
log3 = open('data-log/measure/atta-loss-vs-step-10.txt')
log4 = open('data-log/measure/atta-loss-vs-step-20.txt')
log5 = open('data-log/measure/atta-loss-vs-step-30.txt')
log6 = open('data-log/measure/atta-loss-vs-step-40.txt')


label1 = "m=1"
label2 = "m=5"
label3 = "m=10"
label4 = "m=20"
label5 = "m=30"
label6 = "m=40"

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
length = 40

log_lines1 = log1.readlines()
log_lines2 = log2.readlines()
log_lines3 = log3.readlines()
log_lines4 = log4.readlines()
log_lines5 = log5.readlines()
log_lines6 = log6.readlines()
for i in range(length):
  data1.append([eval(j) for j in log_lines1[i].split(' ')])
  data2.append([eval(j) for j in log_lines2[i].split(' ')])
  data3.append([eval(j) for j in log_lines3[i].split(' ')])
  data4.append([eval(j) for j in log_lines4[i].split(' ')])
  data5.append([eval(j) for j in log_lines5[i].split(' ')])
  data6.append([eval(j) for j in log_lines6[i].split(' ')])

print(len(data1))

x = np.array([i[1] for i in data1])

adv_loss1 = np.array([i[2] for i in data1])
adv_loss2 = np.array([i[2] for i in data2])
adv_loss3 = np.array([i[2] for i in data3])
adv_loss4 = np.array([i[2] for i in data4])
adv_loss5 = np.array([i[2] for i in data5])
adv_loss6 = np.array([i[2] for i in data6])

current_palette = sns.color_palette()

plt.plot(x, adv_loss1, color=current_palette[0], label=label1, lw=2)
plt.plot(x, adv_loss2, color=current_palette[1], label=label2, lw=2)
plt.plot(x, adv_loss3, color=current_palette[2], label=label3, lw=2)
plt.plot(x, adv_loss4, color=current_palette[3], label=label4, lw=2)
plt.plot(x, adv_loss5, color=current_palette[4], label=label5, lw=2)
plt.plot(x, adv_loss6, color=current_palette[5], label=label6, lw=2)
even = [ n for n in x if n%5==0 or n==1]
plt.xticks(even)

plt.xlabel("Attack iterations in each epoch", fontsize=15)
plt.ylabel("Loss value", fontsize=15)
plt.tick_params(labelsize=10)

plt.legend(fontsize='x-large')

plt.savefig('data-pic/num-steps-vs-loss.png')
