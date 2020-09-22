a = [[1,2,3],[5,3,2],[2,6,3]]
b = [[1,0],[1,1]]
c = [[0,0],[0,0]]
padding = [[0,0,0,0,0] for i in range(5)]

for i in range(2):
    for j in range(2):
        s = 0
        for k in range(2):
            for e in range(2):
                s += a[i+k][j+e]*b[k][e]
        c[i][j] = s
print(c)