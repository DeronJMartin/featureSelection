import matplotlib.pyplot as plt
import numpy as np

# For forward selection
# vals = np.loadtxt('plot_val_fw.txt', delimiter=',')
# x = list(range(len(vals[0])))

# For backward elimination
vals = np.loadtxt('plot_val_bw.txt', delimiter=',')
x = list(range(len(vals[0])-1, -1, -1))

y = list(vals[1])

labels = []
l = vals[0]
for i in range(len(l)):
    if l[i] == -1:
        labels.append('None')
    else:
        labels.append(str(int(l[i])))

plt.figure(figsize=(14,7))
plt.plot(x, y, 'bo-')
plt.xlabel("No. of features")
plt.ylabel("Accuracy")
plt.ylim((0,1))
plt.title("Large Dataset 40, Backward Elimination")

for i in range(len(x)):
    plt.annotate(labels[i], (x[i], y[i]), textcoords= "offset points", xytext= (0,-20), ha= 'center')

plt.show()
# plt.savefig("large_bw.png")