import math

def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    """
    Return the nearest power of two to x.
    If round_up=True, choose the next power of two if x isn't already one.
    """
    if round_up:
        return 1 << math.ceil(math.log2(x))
    else:
        return 1 << math.floor(math.log2(x))
