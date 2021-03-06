image = [[3,2,4,5,6],
         [4,5,6,1,2],
         [2,3,1,0,2],
         [4,2,1,4,3],
         [3,4,7,2,1],
]

kernel = [[1,1],[0,1]]
feature = [[0,0,0,0,] for i in range(4)]

W = len(image[0])
F = len(kernel[0])
P = 0
S = 1 

print("feature_map_size",((W-F+2*P )/S)+1)

for i in range(4):
    for j in range(4):
        s = 0
        for k in range(2):
            for e in range(2):
                s += image[i+k][j+e]*kernel[k][e]
        feature[i][j] = s
print(feature)

maxpool = [[0,0] for i in range(2)]
pool_featur = [[0,0] for i in range(2)]

for m in range(2):
    for p in range(2):
        init_max = 0
        for fh in range(2):
            for fw in range(2):
                pool_max = feature[m*2+fh][p*2+fw]
                if init_max < pool_max:
                    maxpool[m][p] = pool_max
                    init_max = pool_max
print(maxpool)


def maxpool2D(maxpool, feature, pool_featur):
    for m in range(len(maxpool[0])):
        for p in range(len(maxpool[0])):
            init_max = 0
            for fh in range(len(pool_featur[0])):
                for fw in range(len(pool_featur[0])):
                    pool_max = feature[m*2+fh][p*2+fw]
                    if init_max < pool_max:
                        maxpool[m][p] = pool_max
                        init_max = pool_max
    print(maxpool)
maxpool2D(maxpool, feature, pool_featur)