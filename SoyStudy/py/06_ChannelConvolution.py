# -*- coding: utf-8 -*- 

# image * kernel
# (3, 3, 2) * (2, 2, 2, 3)

# ---> (2, 2, 3) // shape

# 세로 2 가로 2 채널 3 이 나옴
# 왜 그렇게 계산되는지, 이해되면 코딩 짜보기

image_ch1 = [1, 4, 1, 5, 2, 0, 9, 2, 7]
image_ch2 = [3, 2, 5, 2, 1, 3, 4, 5, 0]

kernel_ch11  = [1, 0, 1, 1]
kernel_ch12 = [1, 0, 0, 1]
kernel_ch13 = [1, 1, 0, 0]

kernel_ch21 = [0, 1, 0, 1]
kernel_ch22 = [0, 1, 1, 0]
kernel_ch23 = [0, 0, 1, 0]

# --------------------------------- #
feature1 = [0 for i in range(4)]
feature2 = [0 for i in range(4)]
feature3 = [0 for i in range(4)]

def conv2D(image_ch1, image_ch2, kernel_ch1, kernel_ch2, feature):
    for i in range(2): 
        for j in range(2):
            s = 0
            for k in range(2):
                for e in range(2):
                    s += image_ch1[(i+k)*3+(j+e)] * kernel_ch1[k*2+e]
                    s += image_ch2[(i+k)*3+(j+e)] * kernel_ch2[k*2+e]
            feature[i*2+j] = s
    return feature

feature1 = conv2D(image_ch1, image_ch2, kernel_ch11, kernel_ch21, feature1)
print(feature1)
