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
from pycbc.types import zeros, complex64, complex128, float32, float64, Array
from pycbc import scheme as _scheme
import numpy as _np
import ctypes
from ctypes import POINTER, byref
from pycbc import libutils
from pycbc.libutils import get_ctypes_library

_libtranspose = get_ctypes_library('transpose', [])
if _libtranspose is None:
    raise ImportError

def plan_transpose(size):
    f = _libtranspose.CreateTranspositionPlan
    f.restype = None
    f.argtypes = [ctypes.c_int, POINTER(POINTER(ctypes.c_int)), ctypes.c_int]
    plan = POINTER(ctypes.c_int)()
    f(0, byref(plan), size)
    print plan.contents.value
    return plan

class BaseTransProblem(_mb.MultiBenchProblem):
    def __init__(self, nrows, ncols, inplace = True):
        # We'll do some arithmetic with these, so sanity check first:
        if (nrows < 1) or (ncols < 1):
            raise ValueError("size must be >= 1")
        if (nrows != ncols):
            raise ValueError("This routine only supports square transpose")
        self.nrows = nrows
        self.ncols = ncols
        self.inplace = inplace
        self.input = zeros(nrows*ncols, dtype=complex64)
        if inplace:
            self.f = _libtranspose.Transpose
            self.f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        else:
            raise RuntimeError("This transposition routine only supports in-place")
        self.f.restype = None
        self.iptr = self.input.ptr

    def _setup(self):
        self.plan = plan_transpose(self.nrows)

    def execute(self):
        self.f(self.iptr, self.nrows, self.plan)


# And finally the actual classes we might call
class InplaceTransProblem(BaseTransProblem):
    def __init__(self, size):
        super(InplaceTransProblem, self).__init__(nrows = size, ncols = size, inplace = True)


# Padding calculation: 32 byte alignement/ (8 bytes per complex64) = 4
pad = 8

class InplacePaddedTransProblem(BaseTransProblem):
    def __init__(self, size):
        size = size+pad
        super(InplacePaddedTransProblem, self).__init__(nrows = size, ncols = size, inplace = True)
            

_class_dict = { 'inplace' : InplaceTransProblem,
                'inplace_padded' : InplacePaddedTransProblem
               }

colfax_trans_valid_methods = _class_dict.keys()

def parse_colfax_trans_problem(probstring, method='inplace'):
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
