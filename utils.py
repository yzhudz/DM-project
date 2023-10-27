import math
from typing import List


# TODO: normalized L2, add another parameter
def normalized_l2_norm(elements: List[int]):
    value = 0
    for index, element in enumerate(elements):
        value += element ** 2
    return math.sqrt(value)
