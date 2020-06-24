import sys

print(sys.path)

from test_import import p62_import2

p62_import2.sum2()

print('-'*33)

# import complite 출력 안됨
from test_import.p62_import2 import sum2
sum2()
