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
import sys

# Several of the OpenMP based approaches use this
max_chunk = 8192

libfftw3f = _fftw.float_lib

fexecute = libfftw3f.fftwf_execute_dft
fexecute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_libtranspose = get_ctypes_library('transpose', [])
if _libtranspose is None:
    raise ImportError

def plan_transpose(size):
    """
    Use FFTW to create a plan for transposing a nrows x ncols matrix,
    either inplace or not.
    """
    f = _libtranspose.CreateTranspositionPlan
    f.restype = None
    f.argtypes = [ctypes.c_int, POINTER(POINTER(ctypes.c_int)), ctypes.c_int]
    plan = POINTER(ctypes.c_int)()
    f(0, byref(plan), size)
    return plan

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

    fplan = libfftw3f.fftwf_plan_many_dft
    fplan.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int]
    fplan.restype = ctypes.c_void_p

    res = ctypes.c_void_p(fplan(1, nptr, nbatch, vin.ptr, None, 1, nsize+padding,
                                vout.ptr, None, 1, nsize+padding,
                                _fftw.FFTW_BACKWARD, flags))

    del vin
    del vout

    return res

hand_fft_support = """
#include <complex>
#include <fftw3.h>
#include <omp.h>

// *** begin code derived from Colfax International transposition ***
// This code supplements the white paper
//    "Multithreaded Transposition of Square Matrices
//     with Common Code for 
//     Intel Xeon Processors and Intel Xeon Phi Coprocessors"
// available at the following URL:
//     http://research.colfaxinternational.com/post/2013/08/12/Trans-7110.aspx
// You are free to use, modify and distribute this code as long as you acknowledge
// the above mentioned publication.
// (c) Colfax International, 2013


#ifdef __cplusplus
extern "C" {
#endif

void Transpose(double* const A, const int n, const int* const plan);
void CreateTranspositionPlan(const int iPlan, int* & plan, const int n);
void DestroyTranspositionPlan(int* plan);

#ifdef __cplusplus
}
#endif

// *** end code derived from Colfax International transposition ***

static inline void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

int i;
float xr, yr, xi, yi, re, im;

for (i=0; i<N; i += 2){
    xr = a[i];
    xi = a[i+1];
    yr = b[i];
    yi = b[i+1];

    re = xr*yr + xi*yi;
    im = xr*yi - xi*yr;

    c[i] = re;
    c[i+1] = im;
}

}

"""

hand_fft_libs = ['transpose', 'fftw3f', 'fftw3f_omp', 'gomp', 'm']
# The following could return system libraries, but oh well,
# shouldn't hurt anything
hand_fft_libdirs = libutils.pkg_config_libdirs(['fftw3f'])
rpath_list = []
for libdir in hand_fft_libdirs:
    rpath = "-Wl,-rpath="+libdir
    rpath_list.append(rpath)

# The code to do an FFT by hand.  Called using weave. Requires:
#     NJOBS1, NJOBS2, NCHUNK1, and NCHUNK2 to be substituted
#           before compilation
# Input arrays: vin, vout, vscratch
# Input plan arrays: plan1, plan2
# Input plans: tplan1, tplan2
#

hand_fft_mul_code = """
int j;

// First, execute half the FFTs for first phase
#pragma omp parallel for schedule(guided, 1)
for (j = 0; j < NJOBS1; j++){
   int tid = omp_get_thread_num();
   ccmul((float *) &aa[j*2*NCHUNK1], (float *) &bb[j*2*NCHUNK1],
         (float *) &vin[j*2*NCHUNK1], 2*NCHUNK1);
   fftwf_execute_dft((fftwf_plan) plan1[tid], (fftwf_complex *) &vin[j*NCHUNK1],
                     (fftwf_complex *) &vout[j*NCHUNK1]);
}

// Next, transpose (really need a twiddle before but we're just
// ballparking)

Transpose((double *) vout, NSIZE, (int *) tplan1[0]);

// Again, parallel
#pragma omp parallel for schedule(static, 1)
for (j = 0; j < NJOBS2; j++){
   int tid = omp_get_thread_num();
   fftwf_execute_dft((fftwf_plan) plan2[tid], (fftwf_complex *) &vout[j*NCHUNK2],
                     (fftwf_complex *) &vout[j*NCHUNK2]);
}

// Finally, transpose again

Transpose((double *) vout, NSIZE, (int *) tplan2[0]);

"""


def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

PAD_LEN = 8

class BaseHandFFTProblem(_mb.MultiBenchProblem):
    def __init__(self, size, padding = True):
        # We'll do some arithmetic with these, so sanity check first:
        if not check_pow_two(size):
            raise ValueError("Only power-of-two sizes supported")

        self.ncpus = _scheme.mgr.state.num_threads
        self.size = size
        self.nfirst = 2 ** int(_np.log2( size ) / 2)
        self.nsecond = size/self.nfirst

        if self.nfirst != self.nsecond:
            raise ValueError("Only supporting perfect square size at the moment")

        if padding:
            self.padding = PAD_LEN
        else:
            self.padding = 0

        self.vsize = (self.nfirst + self.padding) * (self.nsecond + self.padding)
        self.invec = zeros(self.vsize, dtype = complex64)
        self.outvec = zeros(self.vsize, dtype = complex64)
        self.stilde = zeros(self.vsize, dtype = complex64)
        self.htilde = zeros(self.vsize, dtype = complex64)
        # Pointers are probably 64 bits; leave enough space just in case
        self.firstplans = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.secondplans = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.tplan1 = _np.zeros(1, dtype = _np.uintp)
        self.tplan2 = _np.zeros(1, dtype = _np.uintp)
        self.nbatch1 = max_chunk/self.nfirst
        self.nbatch2 = max_chunk/self.nsecond

    def _setup(self):
        # Our transposes are executed using all available threads, and always in-place
        self.tplan1[0] = plan_transpose(self.nsecond + self.padding).value
        self.tplan2[0] = plan_transpose(self.nsecond + self.padding).value
        # Our batched FFTs are executed using a single thread, since they will be called
        # from inside an OpenMP parallel region. We make a plan for each cpu, so that
        # there is not contention to read the plans (and they can stay in cache).
        for i in range(0, self.ncpus):
            self.firstplans[i] = plan_batched(self.nfirst, self.nbatch1, self.padding, 
                                              inplace = False, nthreads = 1).value
            self.secondplans[i] = plan_batched(self.nsecond, self.nbatch2, self.padding, 
                                               inplace = True, nthreads = 1).value
        # Force compilation as part of setup
        self.execute()


# Now our derived classes

class HandFFTMulProblem(BaseHandFFTProblem):
    def __init__(self, size, inplace = False, padding = True):
        super(HandFFTMulProblem, self).__init__(size, inplace = inplace, padding = padding)
        tmpcode = hand_fft_mul_code.replace('NJOBS1', str( self.size/(2*max_chunk) ) )
        tmpcode = tmpcode.replace('NJOBS2', str( self.size/max_chunk ) )
        tmpcode = tmpcode.replace('NCHUNK1', str( (self.nfirst + self.padding) * self.nbatch1))
        tmpcode = tmpcode.replace('NSIZE', str(self.nfirst) )
        self.code = tmpcode.replace('NCHUNK2', str( (self.nsecond + self.padding) * self.nbatch2))
        del tmpcode

    def execute(self):
        aa = _np.array(self.stilde.data, copy = False)
        bb = _np.array(self.htilde.data, copy = False)
        vin = _np.array(self.invec.data, copy = False)
        vout = _np.array(self.outvec.data, copy = False)
        plan1 = _np.array(self.firstplans, copy = False)
        plan2 = _np.array(self.secondplans, copy = False)
        tplan1 = _np.array(self.tplan1, copy = False)
        tplan2 = _np.array(self.tplan2, copy = False)
        inline(self.code, ['aa', 'bb', 'vin', 'vout', 'plan1', 'plan2', 'tplan1', 'tplan2'],
               extra_compile_args=['-fopenmp -march=native -ffast-math -fprefetch-loop-arrays -funroll-loops -O3 -w'],
               libraries = hand_fft_libs, library_dirs = hand_fft_libdirs,
               support_code = hand_fft_support, extra_link_args = rpath_list,
               verbose = 2, auto_downcast = 1)


_class_dict = { 'hand_fft_mul' : HandFFTMulProblem
              }

custom_valid_methods = _class_dict.keys()

def parse_custom_problem(probstring, method):
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
