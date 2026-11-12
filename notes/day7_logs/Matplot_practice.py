import matplotlib.pyplot as plt
from torch import alpha_dropout
import numpy as np

'单折线图绘制'
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 3, 6, 5]

# fig, axes = plt.subplots(1, 1, figsize=(6, 4))

# axes.plot(x ,y)
# axes.set_title('Line Plot')
# axes.set_xlabel('X')
# axes.set_ylabel('Y')
# axes.grid(alpha=0.3)

# plt.show()


confmat = np.array([
    [90, 5, 2],
    [8, 80, 7],
    [3, 6, 91],
])

fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(confmat, cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Confusion Matrix")

plt.show()