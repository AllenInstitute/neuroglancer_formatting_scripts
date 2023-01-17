def get_prime_factors(value):

    result = []
    next_factor = 2
    while True:
        next_factor = _next_prime_factor(
                           current_value=value,
                           starting_factor=next_factor)

        if next_factor is None:
            if value != 1:
                result.append(value)
            return result
        result.append(next_factor)
        value = value // next_factor

    return result


def _next_prime_factor(
        current_value,
        starting_factor):

    factor = starting_factor
    while current_value % factor != 0:
        factor += 1
        if factor**2 > current_value:
            return None
    return factor
