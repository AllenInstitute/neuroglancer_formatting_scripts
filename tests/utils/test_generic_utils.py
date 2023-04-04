from neuroglancer_interface.utils.utils import get_prime_factors

def test_get_prime_factors():

    val = 2*7*11*13*2
    factors = get_prime_factors(val)
    assert factors == [2, 2, 7, 11, 13]

    val = 2*5*3*3*3*5*11*23*23
    factors = get_prime_factors(val)
    assert factors == [2, 3, 3, 3, 5, 5, 11, 23, 23]
