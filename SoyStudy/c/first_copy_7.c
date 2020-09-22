#include <stdio.h>

int add(int x, int y);

int main()
{
    int a = 3;
    int b = 4;

    int c = add(a, b);

    printf("%d\n", c);

    return 0;
}

int add(int x, int y){
    return x + y;
}