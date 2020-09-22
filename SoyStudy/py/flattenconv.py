image = [1,2,3,5,3,2,2,6,3]
kernel = [1,0,1,1]
feature = [0,0,0,0]

# feature = [9, 7, 13, 12]

# def conv(image, kernel, image_len, kernel_len):

#     return c;

for i in range(2): 
    for j in range(2):
        s = 0
        for k in range(2):
            for e in range(2):
                s += image[(i+k)*3+(j+e)] * kernel[k*2+e]
        feature[i*2+j] = s
print(feature)