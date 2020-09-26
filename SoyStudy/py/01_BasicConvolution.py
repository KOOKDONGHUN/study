a = [[1,2,3],[5,3,2],[2,6,3]] # 3,3 # 5,5
b = [[1,0],[1,1]] # 2,2 # 3,3 
c = [[0,0],[0,0]] # 2,2

for i in range(2):
    for j in range(2):
        s = 0
        for k in range(2):
            for e in range(2):
                s += a[i+k][j+e]*b[k][e]
        c[i][j] = s

print(c)

def basix_Conv2D(data, kernel, featur_map, stride=1):
    # feature_map_size = len(data[0]) - len(kennel_size[0]) 
    for i in range(len(featur_map[0])):
        for j in range(len(featur_map[0])):
            s = 0
            for k in range(len(kernel[0])):
                for e in range(len(kernel[0])):
                    s += a[i+k][j+e]*b[k][e]
            c[i][j] = s
    print(c)
    # return feature_map

basix_Conv2D(a, b, c)