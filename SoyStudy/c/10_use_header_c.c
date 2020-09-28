#include <stdio.h>
#include "09_header.h"

int main()
{
    int a = 3;
    int b = 4;

    int c = mul(a, b);

    printf("%d\n", c);

    return 0;
}
// add와 mul이 선언된 헤더 파일을 가져오고 정의된 파일을 같이 컴파일 하거나 메인에서 정의 해줘야 함