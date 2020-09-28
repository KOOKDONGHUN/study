# padding stride 1차원 배열로 구현

image = [0, 0, 0, 0, 0, 0, 0, 3, 2, 4, 5, 0, 0, 2, 3, 1, 0, 0, 0, 4, 2, 1, 4, 0, 0, 3, 4, 7, 2, 0, 0, 0, 0, 0, 0, 0]    
## image가 이미 패딩처리되서 나옴

## image size = (4, 4), padding = (1, 1), stride = (3, 3)
## if stride = (2,2) ...?
feature = [0 for i  in range(4)] ## feature map = (2, 2) // if stride = (2, 2), size?
         

kernel = [1, 1, 0, 0, 1, 0, 1, 0, 1] ## kernel size (3,3)

stride = 3
def conv_strid(image, feature, kernel, stride):

    image_w = int(len(image)**0.5)
    kernel_size = int(len(kernel)**0.5)
    feat_w = int(len(feature)**0.5)

    # print(image_w)        6
    # print(kernel_size)    3
    # print(feat_w)         2


    # tensorflow와 pytorch의 방식이 다르다 

    # print(feature)         [0, 0, 0, 0] 


    for f_h in range(feat_w):
        for f_w in range(feat_w):
            sum = 0
            for k_h in range(kernel_size):
                for k_w in range(kernel_size):
                    sum += image[stride * (f_h + f_w) + image_w * k_h + k_w] * kernel[ stride * k_h + k_w]
                    # print(3*k_h+k_w) 0,1,2,3,4,5,6,7,8
            feature[feat_w * f_h + f_w] = sum
    return feature

print(conv_strid(image, feature, kernel,stride))# [6, 6, 6, 7]


# print(feature)  [6, 6, 6, 7]