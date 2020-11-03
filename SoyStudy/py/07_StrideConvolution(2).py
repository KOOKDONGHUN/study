image = [[0,0,0,0,0,0],
         [0,3,2,4,5,0],
         [0,2,3,1,0,0],
         [0,4,2,1,4,0],
         [0,3,4,7,2,0],
         [0,0,0,0,0,0]]

kernel = [[1,1,0],[0,1,0],[1,0,1]]

feature = [[0,0,0] for i in range(3)]

stride = 2
for i in range(3):
    for j in range(3):
        s = 0
        for k in range(3):

            for e in range(3):

                s += image[i*stride+k][j*stride+e]*kernel[k][e]
                
        feature[i][j] = s
print(feature)