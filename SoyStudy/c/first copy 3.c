#include <stdio.h>

int main()
{
    for (int i = 0; i < 10; i++) {
        if (i % 2){
            printf("%d\n", i);
        }
        else {
            printf("odd\n");
        }
    }
    return 0;
}