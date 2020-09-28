#include <stdio.h>

// 선언만 해줌
int add(int x, int y);
int mul(int x, int y);

// 여기서 정의 해주거나 정의된 c파일을 같이 컴파일 해줘야함 

int main()
{
    int a = 3;
    int b = 4;

    int c = mul(a, b);

    printf("%d\n", c);

    return 0;
}