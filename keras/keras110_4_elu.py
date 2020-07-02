import numpy as np
import matplotlib.pyplot as plt

# def elu(x):
#     if(x>0):
#         return x
#     if(x<0):
#         return 0.2*(np.exp(-x)-1)

def elu(x):
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 0.2*(np.exp(x)-1)
        y_list.append(y)
    return y_list  

x = np.arange(-5,5,0.1)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()