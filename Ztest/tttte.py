img_ch1 = [1, 4, 1, 5, 2, 0, 9, 2, 7]
img_ch2 = [3, 2, 5, 2, 1, 3, 4, 5, 0]

kernel_ch11  = [1, 0, 1, 1]
kernel_ch21 = [0, 1, 0, 1]
kernel_ch12 = [1, 0, 0, 1]
kernel_ch22 = [0, 1, 1, 0]
kernel_ch13 = [1, 1, 0, 0]
kernel_ch23 = [0, 0, 1, 0]

feat_im1 = [0,0,0,0]
feat_im2 = [0,0,0,0]
feat_im3 = [0,0,0,0]

img = [img_ch1, img_ch2]
kernels = [kernel_ch11, kernel_ch21, kernel_ch12, kernel_ch22, kernel_ch13, kernel_ch23]

feat_im = [feat_im1, feat_im2, feat_im3]

print(len(img))



def convnet_ham(img, kernels,stride):
    img_len = len(img[0]) # 9
    k_len = len(kernels[0]) # 4
    # print(img_len) 9
    # print(k_len) 4

    input_ch = len(img) # 2
    output_ch = int(len(kernels)/input_ch)  # 3
    # channels_size = int(len(kernels)/len(img)) # 3
    
    img_len = int(img_len**0.5) #
    k_len = int(k_len**0.5)
    feat_len = int(((img_len - k_len) / stride) + 1)
    # print(channels_size)

    features = [[0 for j in range(feat_len**2)] for i in range(output_ch)]
    # print(features)
    
    for f_h in range(feat_len):

        for f_w in range(feat_len):

            for feat_num in range(len(features)):
                sum = 0
                # sum이 이 포문 바깥에서 선언이 되면 그 이전 값이 저장되어서 다음 필터에 전달 됨
                
                feature = features[feat_num]
                # output 값인 feature의 한 채널값만 가져온다

                for img_num in range(len(img)):
                    img_ch = img[img_num]
                    # 인풋 이미지 채널에서 한 이미지의 채널만 가져옴

                    kernel = kernels[input_ch*feat_num + img_num]
                    # input_ch(인풋 채널) 갯수만큼 kernels들이(2차원 배열에서 행으로 분리되어 있음) 나눠지며 
                    # img_num(이미지)갯수만큼 kernel과 img_ch가 1:1 매칭되어 conv연산진행

                    for k_h in range(k_len):
                        for k_w in range(k_len):
                            sum += img_ch[img_len*(f_h+k_h)+f_w+k_w]*kernel[k_len*k_h+k_w]
                            # 이 문제의 경우 kernel이 두번씩 즉 sum이 두개 필요하지만 여기서는 kernel단에서 for문이 두번 돌기에 상관 없음
                feature[feat_len*f_h + f_w] = sum
                # print(features)
                # output_channel인 feat_num에서 out_channel 갯수만큼 for문이 돌아짐
                # [[11, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                # [[11, 0, 0, 0], [7, 0, 0, 0], [0, 0, 0, 0]]
                # [[11, 0, 0, 0], [7, 0, 0, 0], [7, 0, 0, 0]]
                # [[11, 14, 0, 0], [7, 0, 0, 0], [7, 0, 0, 0]]
                # [[11, 14, 0, 0], [7, 10, 0, 0], [7, 0, 0, 0]]
                # [[11, 14, 0, 0], [7, 10, 0, 0], [7, 6, 0, 0]]
                # [[11, 14, 22, 0], [7, 10, 0, 0], [7, 6, 0, 0]]
                # [[11, 14, 22, 0], [7, 10, 12, 0], [7, 6, 0, 0]]
                # [[11, 14, 22, 0], [7, 10, 12, 0], [7, 6, 11, 0]]
                # [[11, 14, 22, 14], [7, 10, 12, 0], [7, 6, 11, 0]]
                # [[11, 14, 22, 14], [7, 10, 12, 17], [7, 6, 11, 0]]
                # [[11, 14, 22, 14], [7, 10, 12, 17], [7, 6, 11, 7]]
                # 이런식으로 숫자들이 채워짐
    return features
    
print(convnet_ham(img, kernels, 1))

# [11, 14, 22, 14]
# [7, 10, 12, 17]
# [7, 6, 11, 7]