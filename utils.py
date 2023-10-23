import math
from typing import List


def normalized_l2_norm(elements: List[int], normalized_array: List[int]):
    value = 0
    for index, element in enumerate(elements):
        if normalized_array[index] != 0:
            value += (element / normalized_array[index]) ** 2
    return math.sqrt(value)
