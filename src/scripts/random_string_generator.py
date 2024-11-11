import random
import string

random.seed(42)


def get_random_string_generator(length=10, postfix=None, prefix=None):
    base = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))
    if prefix is not None:
        base = prefix+base

    if postfix is not None:
        base += postfix

    return base
