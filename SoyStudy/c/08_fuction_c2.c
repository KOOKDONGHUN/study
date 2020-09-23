#include <stdio.h>

int mul(int x, int y);

int main()
{
    int a = 3;
    int b = 4;

    int c = mul(a, b);

    printf("%d\n", c);

    return 0;
}

int mul(int x, int y){
    return x * y;
}