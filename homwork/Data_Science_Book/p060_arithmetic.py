from typing import List
import math

Vec = List[float]

height_weight_age = [70, # 인치
                     170, # 파운드
                     40 # 나이
]

grades = [95, # 시험1 점수
          80, # 시험2 점수
          75, # 시험3 점수
          62  # 시험4 점수
]

''' 벡터의 덧셈은  zip를 사용해서 두 벡터를 묶은 뒤, 각 성분끼리 더하는 리스트 컴프리헨션을 적용한다. '''

def add(v='vector', w='vector') :
    """ 각 성분끼리 더한다. """
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i,w_i in zip(v,w)]
assert add([1,2,3],[4,5,6]) == [5,7,9] # 아무것도 출력되지 않으면 정상작동

def subtract(v='vector',w='vector'):
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i,w_i in zip(v,w)]

assert subtract([5,7,9],[4,5,6]) == [1,2,3]

def vector_sum(vectors) :
    """ 모든 벡터의 각 성분들끼리 더한다. """
    # vectors가 비어있는지 확인
    assert vectors, "no vectors provied!"

    # 모든 벡터의 길이가 동일한지 확인
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # i번째 결과 값은 모든 벡터의 i번쨰 성분을 더한 값
    return [sum(vector[i] for vector in vectors)
                for i in range(num_elements)]

assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]

def scalar_multiply(c, v) :
    """ 모든 성분을 c로 곱하기."""

    return [c* v_i for v_i in v]

assert scalar_multiply(2,[1,2,3]) == [2, 4, 6]

""" 벡터의 내적(dot product) : 각 성분별 곱한 값을 더해준 값 """
def dot(vlist, wlist):
    assert len(vlist) == len(wlist), "vectors must be same length"

    return sum(v * w for v,w in zip(vlist,wlist))

assert dot([1,2,3],[4,5,6]) == 32 # 1*4 + 2*5 + 3*6

""" 내적의 개념을 사용하면 각 성분의 제곱 값의 합을 쉽게 구할 수 있다. """

def sum_of_squares(v):
    return dot(v,v)

assert sum_of_squares([1,2,3]) == 14 # 1*1 + 2*2 + 3*3

def magnitude(vlist):
    return math.sqrt(sum_of_squares(v))

assert magitude([3,4]) == 5