import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
print(dir(fig))
fig.axes[0].set_title(r'is Number $\sum_{n=1}^\infty$')
print(fig.axes)
plt.show()
