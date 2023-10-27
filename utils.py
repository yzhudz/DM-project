import math
from typing import List


def normalized_l2_norm(elements: List[int], a_value: List[int]):
    value = 0
    for index, element in enumerate(elements):
        if a_value[index] != 0:
            value += (element / a_value[index]) ** 2
    return math.sqrt(value)
