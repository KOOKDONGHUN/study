#include <stdio.h>
#include <stdlib.h>

int main() {
    // int* image(int*)malloc(5*sizeof(int));
    int image[5][5] = {{3,2,4,5,6},
                        {4,5,6,1,2},
                        {2,3,1,0,2},
                        {4,2,1,4,3},
                        {3,4,7,2,1},
                    };

    int kernel[2][2] = {{1,1},
                        {0,1},
                    };

    int feature[4][4] = {{0,0,0,0},
                         {0,0,0,0},
                         {0,0,0,0},
                         {0,0,0,0},
                    };
    
    int maxpool[2][2] = {{0,0},
                         {0,0}
                    };

    int s = 0;
    int init_max = 0;
    int pool_max = 0;


    for (int i = 0 ; i < 4 ; i++){
        for (int j = 0 ; j < 4 ; j++){
            s = 0;
            for (int k = 0 ; k < 2 ; k++){
                for (int e = 0 ; e < 2 ; e++){
                    s += image[i+k][j+e]*kernel[k][e];
                }
            feature[i][j] = s;
            }
            printf("%d, %d :: %d\n", i,j,feature[i][j]);
        }
    }

printf("========================================================\n");
    for (int m = 0 ; m < 2 ; m++){
        for (int p = 0 ; p < 2 ; p++){
            init_max = 0;
            for (int fh = 0 ; fh < 2 ; fh++){
                for (int fw = 0 ; fw < 2 ; fw++){
                    pool_max = feature[m*2+fh][p*2+fw];
                    if (init_max < pool_max) {
                        maxpool[m][p] = pool_max;
                        init_max = pool_max;
                    }
                }
            }
            printf("%d, %d :: %d\n", m, p, maxpool[m][p]);
        }
    }
}





