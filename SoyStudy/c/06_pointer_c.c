#include <stdio.h>
#include <stdlib.h>

int main(){
    int* a;

    *a = 3;
    
    printf("%p\n",a);
    printf("%d\n",*a);

    return 0;
}