# import p31_sample
from p31_sample import test

x = 222

def main_func():
    print(f'x : {x}')

# p31_sample.test() # x : 111
test() # x : 111

main_func() # x : 222

