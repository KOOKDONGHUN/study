image = [1,2,3,5,3,2,2,6,3]
kernel = [1,0,1,1]
feature = [0,0,0,0]

# feature = [9, 7, 13, 12]

# def conv(image, kernel, image_len, kernel_len):

#     return c;

# Tensor size or shape: (width = 28, height = 28)
# Convolution filter size (F): (F_width = 5, F_height = 5)
# Padding (P): 0
# Stride (S): 1
# Using the equation:
# output width=((W-F+2*P )/S)+1

W = len(image)**0.5
F = len(kernel)**0.5
P = 0
S = 1 

print(((W-F+2*P )/S)+1)

for i in range(2): 
    for j in range(2):
        s = 0
        for k in range(2):
            for e in range(2):
                s += image[(i+k)*3+(j+e)] * kernel[k*2+e]
        feature[i*2+j] = s
print(feature)

# def conv(image, kernel, image_len=len(image)**0.5, kernel_len=len(kernel)**0.5):
#     for i in range()
#     return 0

