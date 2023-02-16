import math

# Cantor encoding
# https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
def cantor_encode(x, y):
    if x < 0 or y < 0:
        raise ValueError(f"{x} and {y} cannot be paired due to negative values")
    z = int(0.5 * (x + y) * (x + y + 1) + y)
    if (x, y) != cantor_decode(z):
        raise ValueError(f"{x} and {y} cannot be paired due to large number")
    return z


# Cantor decoding
# https://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
def cantor_decode(z):
    w = math.floor((math.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    return x, y