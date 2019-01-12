import matplotlib.pyplot as plt
import numpy as np

X0 = np.array([[0.0, 0.0],
               [0.2, 0.3],
               [0.1, 0.1],
               [0.3, 0.2],
               [0.5, 1.0]])

X1 = np.array([[0.0, 1.0],
               [0.2, 0.7],
               [0.1, 0.9],
               [0.3, 0.8]])

plt.scatter(X0[:,0], X0[:,1], c='red')
plt.scatter(X1[:,0], X1[:,1], c='blue')

plt.plot([0.4, 0.4], [-0.1, 1.1], c='black')
plt.plot([-0.1, 0.4], [0.4, 0.4], c='black')

plt.xlim(-0.05,0.55)
plt.ylim(-0.05,1.1)

plt.xlabel('x0')
plt.ylabel('x1')

plt.show()
