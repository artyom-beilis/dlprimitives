"""
Main module for dlprim library, imports all classes from two modules:

- _pydlprim - Boost.Python wrapper of C++ dlprim library - all its classes imported into dlprim 
- dlprim.netconfig - helper functions to work with dlprim library
"""

from ._pydlprim import *
from .netconfig import *

