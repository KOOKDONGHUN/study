#include <stdio.h>
#include <stdlib.h>

void add_array(int* z, int* x, int* y, int length){
    for (int i = 0; i < length ; i++){
        z[i] = x[i] + y[i];
    }
}

// int* add_array(int* x, int* y, int length){
//     int* z = (int*)malloc(length*sizeof(int)); # 메인에서 포인터를 관리 할 수 있도록
//     for (int i = 0; i < length ; i++){
//         z[i] = x[i] + y[i];
//     }
//     return z;
// }

int main(){
    int* a = (int*)malloc(5*sizeof(int));
    int* b = (int*)malloc(5*sizeof(int));

    for (int i = 0; i < 5; i++) {
        a[i] = 2*i;
    }

    for (int i = 0; i < 5; i++) {
        b[i] = 2*i + 1;
    }

    int* c = (int*)malloc(5*sizeof(int));

    add_array(c, a, b, 5);

    for (int i = 0 ; i < 5 ; i++){
        printf("%d\n", c[i]);
    }

    // 할당 된 메모리를 반납
    free(a);
    free(b);
    free(c);

    return 0;
}