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

W = len(image)**0.5
F = len(kernel)**0.5
P = 0
S = 1 

print(((W-F+2*P )/S)+1)

def conv(image, kernel, image_len=int(len(image)**0.5), kernel_len=int(len(kernel)**0.5), stride=1, padding=0):

    W = int(len(image)**0.5)
    F = int(len(kernel)**0.5)
    P = padding
    S = stride

    feature_map_size = int(( ( W - F + 2 * P ) / S ) + 1)

    feature = [ 0 for i in range(int(feature_map_size)**2)]

    for i in range(feature_map_size):
        for j in range(feature_map_size):
            v = 0
            for k in range(kernel_len):
                for e in range(kernel_len):
                    v += image[(i+k)*3+(j+e)] * kernel[k*2+e]
            feature[i*2+j] = v

    return feature

obj = conv(image,kernel)
print(obj)