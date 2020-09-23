#include <stdio.h>
#include <stdlib.h>

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

    for (int i = 0; i < 5; i++) {
        c[i] = a[i] + b[i];
        printf("%d\n", c[i]);
    }

    return 0;
}