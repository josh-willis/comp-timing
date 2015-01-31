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
#from pycbc.fft.fftw_pruned import plan_transpose, fft_transpose_fftw
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

def plan_batched(nsize, nbatch=1, padding=0, inplace=False, nthreads=1):
    """
    A function to create an FFTW plan for a batched, strided,
    possibly multi-threaded transform.
    """
    if not _fftw._fftw_threaded_set:
        _fftw.set_threads_backend()
    if nthreads != _fftw._fftw_current_nthreads:
        _fftw._fftw_plan_with_nthreads(nthreads)
    # Convert a measure-level to flags
    flags = _fftw.get_flag(_fftw.get_measure_level(), aligned=True)

    narr = _np.zeros(1, dtype=_np.int32)
    narr[0] = nsize
    nptr = narr.ctypes.data

    N = (nsize+padding)*nbatch
    vin = zeros(N, dtype = complex64)
    if inplace:
        vout = vin
    else:
        vout = zeros(N, dtype = complex64)

    f = libfftw3f.fftwf_plan_many_dft
    f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int]
    f.restype = ctypes.c_void_p

    res = f(1, nptr, nbatch, vin.ptr, None, 1, nsize+padding,
            vout.ptr, None, 1, nsize+padding, _fftw.FFTW_BACKWARD, flags)

    del vin
    del vout

    return res

hand_fft_support = """
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
"""

hand_fft_libs = ['fftw3f', 'fftw3f_omp', 'gomp', 'm']
# The following could return system libraries, but oh well,
# shouldn't hurt anything
hand_fft_libdirs = libutils.pkg_config_libdirs(['fftw3f'])

# The code to do an FFT by hand.  Called using weave. Requires:
#     NJOBS1, NJOBS2, NCHUNK1, and NCHUNK2 to be substituted
#           before compilation
# Input arrays: vin, vout, vscratch
# Input plan arrays: plan1, plan2
# Input plans: tplan1, tplan2
#
hand_fft_code = """
int j;

// First, execute half the FFTs for first phase
#pragma omp parallel for schedule(static)
for (j = 0; j < NJOBS1; j++){
   int tid = omp_get_thread_num();
   fftwf_execute_dft(plan1[tid], &vin[j*NCHUNK1], &vscratch[j*NCHUNK1]);
}

// Next, transpose (really need a twiddle before but we're just
// ballparking)

fftwf_execute_dft(tplan1[0], vscratch, vout);

// Again, parallel
#pragma omp parallel for schedule(static)
for (j = 0; j < NJOBS2; j++){
   int tid = omp_get_thread_num();
   fftwf_execute_dft(plan2[tid], &vout[j*NCHUNK2], &vscratch[j*NCHUNK2]);
}

// Finally, transpose again

fftwf_execute_dft(tplan2[0], vscratch, vout);

"""


def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

PAD_LEN = 8

class BaseHandFFTProblem(_mb.MultiBenchProblem):
    def __init__(self, size, inplace = False, padding = True):
        # We'll do some arithmetic with these, so sanity check first:
        if not check_pow_two(size):
            raise ValueError("Only power-of-two sizes supported")
        if not 

        self.ncpus = _scheme.mgr.state.num_threads
        self.size = size
        self.nfirst = 2 ** int(_np.log2( size ) / 2)
        self.nsecond = size/self.nfirst

        if padding:
            self.padding = PAD_LEN
        else:
            self.padding = 0

        self.vsize = (self.nfirst + self.padding) * (self.nsecond + self.padding)
        self.invec = zeros(self.vsize, dtype = complex64)
        self.inplace = inplace
        if inplace:
            self.outvec = self.invec
        else:
            self.outvec = zeros(self.vsize, dtype = complex64)
        # For our transpositions
        self.tmpvec = zeros(self.vsize, dtype = complex64)
        # Pointers are probably 64 bits; leave enough space just in case
        self.firstplans = _np.zeros(self.ncpus, dtype = _np.uint64)
        self.secondplans = _np.zeros(self.ncpus, dtype = _np.uint64)
        self.tplan1 = _np.zeros(1, dtype = _np.uint64)
        self.tplan2 = _np.zeros(1, dtype = _np.uint64)
        self.nbatch1 = max_chunk/self.nfirst
        self.nbatch2 = max_chunk/self.nsecond
        tmpcode = hand_fft_code.replace('NJOBS1', str(self.size/max_chunk))
        tmpcode = tmpcode.replace('NJOBS2', str(self.size/max_chunk))
        tmpcode = tmpcode.replace('NCHUNK1', str( (self.nfirst + self.padding) * self.nbatch1))
        self.code = tmpcode.replace('NCHUNK2', str( (self.nsecond + self.padding) * self.nbatch2))
        del tmpcode

    def _setup(self):
        # Our transposes are executed using all available threads, and always out-of-place
        self.tplan1[0] = plan_transpose(self.nsecond + self.padding, self.nfirst + self.padding,
                                        inplace = False, _scheme.mgr.state.num_threads)
        self.tplan2[0] = plan_transpose(self.nsecond + self.padding, self.nfirst + self.padding,
                                        inplace = False, _scheme.mgr.state.num_threads)
        # Our batched FFTs are executed using a single thread, since they will be called
        # from inside an OpenMP parallel region. We make a plan for each cpu, so that
        # there is not contention to read the plans (and they can stay in cache).
        for i in range(0, self.ncpus):
            self.firstplans[i] = plan_batched(self.nfirst, self.nbatch1, self.padding, 
                                              inplace = False, nthreads = 1)
            self.secondplans[i] = plan_batched(self.nsecond, self.nbatch2, self.padding, 
                                               inplace = False, nthreads = 1)

    def execute(self):
        vin = _np.array(self.invec.data, copy = False)
        vout = _np.array(self.outvec.data, copy = False)
        vscratch = _np.array(self.tmpvec.data, copy = False)
        plan1 = _np.array(self.firstplans.data, copy = False)
        plan2 = _np.array(self.secondplans.data, copy = False)
        tplan1 = _np.array(self.tplan1, copy = False)
        tplan2 = _np.array(self.tplan2, copy = False)
        inline(self.code, ['vin', 'vout', 'vscratch', 'plan1', 'plan2', 'tplan1', 'tplan2'],
               extra_compile_args=['-fopenmp'], libraries = hand_fft_libs, library_dirs = hand_fft_libdirs,
               support_code = hand_fft_support)

# Now our derived classes

_class_dict = { 'hand_fft' : BaseHandFFTProblem
               }

hand_fft_valid_methods = _class_dict.keys()

def parse_hand_fft_problem(probstring, method):
    """
    This function takes a string of the form <number>


    It returns the class and size, so that the call:
        MyClass, n = parse_trans_problem(probstring)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
