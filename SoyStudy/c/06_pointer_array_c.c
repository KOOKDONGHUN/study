#include <stdio.h>
#include <stdlib.h>

int main(){
    int* a = (int*)malloc(5*sizeof(int));

    printf("%d\n",a[0]);
    printf("%p\n",&a[0]);

    return 0;
}