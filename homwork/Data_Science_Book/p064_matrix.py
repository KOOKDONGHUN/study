Matrix = List[List(float)]

A = [[1,2,3],
     [4,5,6]] # A는 2개의 행과 3개의 열로 구성되어 있다.

B = [[1,2],
     [3,4],
     [5,6]] # B는 3개의 행과 2개의 열로 구성되어 있다.

from typing import Tuple

def shape(Amatrix) :
    """(열의 개수,행의 개수)를 반환"""
    num_rows = len(Amatrix)
    num_cols = len(Amatrix[0])
    
    return num_rows, num_cols