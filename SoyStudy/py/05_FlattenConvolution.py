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

W = len(image)**0.5 # 이미지 크기 # image = [1,2,3,5,3,2,2,6,3]
F = len(kernel)**0.5 # 커널 사이즈 # kernel = [1,0,1,1]
P = 0 # 패딩의 유무 0이니까 패딩이 없고 1이면 패딩이 있음 패딩이 위, 아레, 좌, 우 어디로 붙는지는 아직 모름 pytorch하고 tensorflow의 패딩 붙는 방식이 다른다
S = 1 # 스트라이드 커널 사이즈를 옮겨가는 간격

print(((W-F+2*P )/S)+1) # 컨볼루전 레이어 하나를 거치고 난후의 피처맵의 크기

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