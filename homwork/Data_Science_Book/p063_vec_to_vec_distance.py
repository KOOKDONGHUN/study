"""  ______________________________________
    √  (v1 - w1)² + ... + (vｎ- wｎ)²           """
import math
from p060_arithmetic import sum_of_squares,subtract,magnitude

def squared_distance(vlist,wlist):
    return sum_of_squares(subtract(vlist,wlist))

# def distance(vlist,wlist):
#     return math.sqrt(squared_distance(vlist,wlist))
def distance(vlist,wlist):
    return magnitude(subtract(vlist,wlist))
 