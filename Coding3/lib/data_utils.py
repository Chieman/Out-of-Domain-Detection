from collections import Counter
from functools import partialmethod
from typing import List, Optional


def partial_class(cls, *args, **kwargs):
    class PartClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return PartClass