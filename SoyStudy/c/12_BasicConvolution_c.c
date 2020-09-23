#include <stdio.h>

int main()
{
    int a[3][3] = {{1,2,3},{5,3,2},{2,6,3}};
    int b[2][2] = {{1,0},{1,1}};
    int c[2][2] = {{0,0},{0,0}};
    int s = 0;

    for (int i = 0 ; i < 2; i++){
        for (int j = 0 ; j < 2; j++){
            s = 0;
            for (int k = 0 ; k < 2; k++){
                for (int e = 0 ; e < 2; e++){
                    s += a[i+k][j+e]*b[k][e];
                }
            c[i][j] = s;
            }
        }
    }
    for (int q = 0; q < 2; q++){
        for (int w = 0; w < 2; w++){
            printf("%d\n",c[q][w]);
        }
    }
    
    return 0;
}