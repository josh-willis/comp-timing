# Copyright (C) 2014 Josh Willis
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import multibench as _mb
from pycbc.types import zeros, float32, complex128, float32, float64, Array
from pycbc import scheme as _scheme
import numpy as _np
import ctypes
from pycbc import libutils
from pycbc.libutils import get_ctypes_library

_libopenblas = get_ctypes_library('openblas', [])
if _libopenblas is None:
    raise ImportError

# From openblas/cblas.h
CBLAS_ORDER_ROWMAJOR = 101
CBLAS_ORDER_COLMAJOR = 102
CBLAS_TRANS_NOTRANS = 111
CBLAS_TRANS_TRANS = 112
CBLAS_TRANS_CONJTRANS = 113
CBLAS_TRANS_CONJNOTRANS = 114

class BaseTransProblem(_mb.MultiBenchProblem):
    def __init__(self, nrows, ncols, inplace = True):
        # We'll do some arithmetic with these, so sanity check first:
        if (nrows < 1) or (ncols < 1):
            raise ValueError("size must be >= 1")
        self.nrows = nrows
        self.ncols = ncols
        self.inplace = inplace
        _libopenblas.openblas_set_num_threads(_scheme.mgr.state.num_threads)
#        self.alpha = Complex()
#        self.alpha.real = 1.0
#        self.alpha.imag = 0.0
        self.alpha = 1.0
        self.input = zeros(nrows*ncols, dtype=float32)
        if inplace:
            self.output = self.input
#            self.f = _libopenblas.cblas_cimatcopy
            self.f = _libopenblas.cblas_simatcopy
            self.f.argtypes = [ctypes.c_uint, ctypes.c_uint,     # ordering, trans (OPENBLAS enums)
                               ctypes.c_uint, ctypes.c_uint,     # nrows, ncols
                               ctypes.c_float, ctypes.c_void_p,  # alpha (=1.0), matrix_ptr
                               ctypes.c_uint, ctypes.c_uint]     # lda, ldb
        else:
            self.output = zeros(nrows*ncols, dtype=float32)
#            self.f = _libopenblas.cblas_comatcopy
            self.f = _libopenblas.cblas_somatcopy
            self.f.argtypes = [ctypes.c_uint, ctypes.c_uint,     # ordering, trans
                               ctypes.c_uint, ctypes.c_uint,     # nrows, ncols
                               ctypes.c_float,                   # alpha (=1.0)
                               ctypes.c_void_p, ctypes.c_uint,   # input, lda
                               ctypes.c_void_p, ctypes.c_uint]   # output, ldb
        self.f.restype = None
        self.iptr = self.input.ptr
        self.optr = self.output.ptr

    def _setup(self):
        pass

# Now our derived base classes
class BaseInplaceTransProblem(BaseTransProblem):
    def __init__(self, nrows, ncols):
        super(BaseInplaceTransProblem, self).__init__(nrows = nrows, ncols = ncols, inplace = True)

    def execute(self):
        self.f(CBLAS_ORDER_ROWMAJOR, CBLAS_TRANS_TRANS, self.nrows, self.ncols,
               self.alpha, self.iptr, self.ncols, self.nrows)
            
class BaseOutplaceTransProblem(BaseTransProblem):
    def __init__(self, nrows, ncols):
        super(BaseOutplaceTransProblem, self).__init__(nrows = nrows, ncols = ncols, inplace = False)

    def execute(self):
        self.f(CBLAS_ORDER_ROWMAJOR, CBLAS_TRANS_TRANS, self.nrows, self.ncols,
               self.alpha, self.iptr, self.ncols, self.optr, self.nrows)

# And finally the actual classes we might call
class InplaceTransProblem(BaseInplaceTransProblem):
    def __init__(self, size):
        super(InplaceTransProblem, self).__init__(nrows = size, ncols = size)

class OutplaceTransProblem(BaseOutplaceTransProblem):
    def __init__(self, size):
        super(OutplaceTransProblem, self).__init__(nrows = size, ncols = size)


# Padding calculation: 32 byte alignement/ (8 bytes per float32) = 4
pad = 8

class InplacePaddedTransProblem(BaseInplaceTransProblem):
    def __init__(self, size):
        size = size+pad
        super(InplacePaddedTransProblem, self).__init__(nrows = size, ncols = size)
            
class OutplacePaddedTransProblem(BaseOutplaceTransProblem):
    def __init__(self, size):
        size = size+pad
        super(OutplacePaddedTransProblem, self).__init__(nrows = size, ncols = size)

_class_dict = { 'inplace' : InplaceTransProblem,
                'outplace' : OutplaceTransProblem,
                'inplace_padded' : InplacePaddedTransProblem,
                'outplace_padded' : OutplacePaddedTransProblem
               }

openblas_trans_valid_methods = _class_dict.keys()

def parse_openblas_trans_problem(probstring, method='inplace'):
    """
    This function takes a string of the form <number>

    It also takes another argument, a string indicating which class
    type to return. 

    It returns the class and size, so that the call:
        MyClass, n = parse_trans_problem(probstring, method)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
