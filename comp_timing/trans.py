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
from pycbc.fft.fftw_pruned import plan_transpose, fft_transpose_fftw
import pycbc.fft
import pycbc.fft.fftw as _fftw
from pycbc import scheme as _scheme
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils
from math import sqrt

# Several of the OpenMP based approaches use this
#max_chunk = 4096
max_chunk = 8192
#max_chunk = 131072
#max_chunk = 65536
#max_chunk = 32768

libfftw3f = _fftw.float_lib

fexecute = libfftw3f.fftwf_execute_dft
fexecute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def plan_transpose(nrows, ncols, inplace, nthreads):
    """
    Use FFTW to create a plan for transposing a nrows x ncols matrix,
    either inplace or not.
    """
    if not _fftw._fftw_threaded_set:
        _fftw.set_threads_backend()
    if nthreads != _fftw._fftw_current_nthreads:
        _fftw._fftw_plan_with_nthreads(nthreads)
    # Convert a measure-level to flags
    flags = _fftw.get_flag(_fftw.get_measure_level(), aligned=True)

    iodim = _np.zeros(6, dtype=_np.int32)
    iodim[0] = nrows
    iodim[1] = 1
    iodim[2] = ncols
    iodim[3] = ncols
    iodim[4] = nrows
    iodim[5] = 1

    N = nrows*ncols

    vin = zeros(N, dtype=complex64)
    if inplace:
        vout = vin
    else:
        vout = zeros(N, dtype=complex64)

    f = libfftw3f.fftwf_plan_guru_dft
    f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                  ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_int]
    f.restype = ctypes.c_void_p

    res = f(0, None, 2, iodim.ctypes.data, vin.ptr, vout.ptr, None, flags)

    del vin
    del vout

    return res

class BaseTransProblem(_mb.MultiBenchProblem):
    def __init__(self, nrows, ncols, inplace = True):
        # We'll do some arithmetic with these, so sanity check first:
        if (nrows < 1) or (ncols < 1):
            raise ValueError("size must be >= 1")
        self.nrows = nrows
        self.ncols = ncols
        self.inplace = inplace
        self.input = zeros(nrows*ncols, dtype=complex64)
        if inplace:
            self.output = self.input
        else:
            self.output = zeros(nrows*ncols, dtype=complex64)
        self.iptr = self.input.ptr
        self.optr = self.output.ptr

    def _setup(self):
        self.plan = plan_transpose(self.nrows, self.ncols, self.inplace,
                                   _scheme.mgr.state.num_threads)

    def execute(self):
        fexecute(self.plan, self.iptr, self.optr)

# Now our derived classes
class InplaceTransProblem(BaseTransProblem):
    def __init__(self, size):
        super(InplaceTransProblem, self).__init__(nrows = size, ncols = size, inplace = True)
            
class OutplaceTransProblem(BaseTransProblem):
    def __init__(self, size):
        super(OutplaceTransProblem, self).__init__(nrows = size, ncols = size, inplace = False)

# Padding calculation: 32 byte alignement/ (8 bytes per complex64) = 4
pad = 8

class InplacePaddedTransProblem(BaseTransProblem):
    def __init__(self, size):
        size = size+pad
        super(InplacePaddedTransProblem, self).__init__(nrows = size, ncols = size, inplace = True)
            
class OutplacePaddedTransProblem(BaseTransProblem):
    def __init__(self, size):
        size = size+pad
        super(OutplacePaddedTransProblem, self).__init__(nrows = size, ncols = size, inplace = False)

_class_dict = { 'inplace' : InplaceTransProblem,
                'outplace' : OutplaceTransProblem,
                'inplace_padded' : InplacePaddedTransProblem,
                'outplace_padded' : OutplacePaddedTransProblem
               }

trans_valid_methods = _class_dict.keys()

def parse_trans_problem(probstring, method='inplace'):
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
