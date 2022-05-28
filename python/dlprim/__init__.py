###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
"""
Main module for dlprim library, imports all classes from two modules:

- _pydlprim - Boost.Python wrapper of C++ dlprim library - all its classes imported into dlprim 
- dlprim.netconfig - helper functions to work with dlprim library
"""

from ._pydlprim import *
from .netconfig import *


def _all_of(mod):
    try:
        yield from mod.__all__
        return
    except AttributeError:
        pass
    for item in dir(mod):
        if not item.startswith('_'):
            yield item

__all__ = [*_all_of(_pydlprim), *_all_of(netconfig)]

