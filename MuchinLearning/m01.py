import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) 
y = np.sin(x)

plt.plot(x, y) # 0부터10까지 0.1씩 증가하는데 그것에 대한 sin값을 보는 그래프

plt.show()