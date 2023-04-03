from typing import Optional, Union
import os
import pathlib
import tempfile


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


def _clean_up(target_path):
    target_path = pathlib.Path(target_path)
    if target_path.is_file():
        target_path.unlink()
    elif target_path.is_dir():
        for sub_path in target_path.iterdir():
            _clean_up(sub_path)
        target_path.rmdir()


def mkstemp_clean(
        dir: Optional[Union[pathlib.Path, str]] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None) -> str:
    """
    A thin wrapper around tempfile mkstemp that automatically
    closes the file descripter returned by mkstemp.

    Parameters
    ----------
    dir: Optional[Union[pathlib.Path, str]]
        The directory where the tempfile is created

    prefix: Optional[str]
        The prefix of the tempfile's name

    suffix: Optional[str]
        The suffix of the tempfile's name

    Returns
    -------
    file_path: str
        Path to a valid temporary file

    Notes
    -----
    Because this calls tempfile mkstemp, the file will be created,
    though it will be empty. This wrapper is needed because
    mkstemp automatically returns an open file descriptor, which was
    causing some of our unit tests to overwhelm the OS's limit
    on the number of open files.
    """
    (descriptor,
     file_path) = tempfile.mkstemp(
                     dir=dir,
                     prefix=prefix,
                     suffix=suffix)

    os.close(descriptor)
    return file_path
