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
from pycbc.filter.matchedfilter_cpu import correlate_numpy, correlate_inline
try:
    from pycbc.filter.matchedfilter_cpu import correlate_simd
except:
    correlate_simd = None
from pycbc.filter.matchedfilter_cpu import omp_flags, omp_libs
from pycbc.filter.matchedfilter_cpu import support as omp_support
from pycbc import scheme as _scheme
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils


class BaseCorrProblem(_mb.MultiBenchProblem):
    def __init__(self, size, dtype):
        # We'll do some arithmetic with these, so sanity check first:
        if (size < 1):
            raise ValueError("size must be >= 1")
        if dtype not in [complex64, complex128]:
            raise ValueError("Correlation only supports complex types")

        self.x = zeros(size, dtype=dtype)
        self.y = zeros(size, dtype=dtype)
        self.z = zeros(size, dtype=dtype)

# Several of the OpenMP based approaches use this
max_chunk = 8192
#max_chunk = 131072
#max_chunk = 4095

# Now our derived classes
class NumpyCorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        super(NumpyCorrProblem, self).__init__(size=size, dtype=dtype)
            
    def execute(self):
        correlate_numpy(self.x, self.y, self.z)

    def _setup(self):
        pass

class WeaveCorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        super(WeaveCorrProblem, self).__init__(size=size, dtype=dtype)

    def execute(self):
        correlate_inline(self.x, self.y, self.z)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

class NewPyCBCCorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        if correlate_simd is None:
            raise RuntimeError("Your version of PyCBC does not have correlate_simd")
        if size != 1048576:
            raise RuntimeError("Current hacky version requires size = 1048576")
        if dtype is not complex64:
            raise RuntimeError("Current hacky version requires single precision")
        super(NewPyCBCCorrProblem, self).__init__(size=size, dtype=dtype)

    def execute(self):
        correlate_simd(self.x, self.y, self.z)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

# From openblas/cblas.h:

CblasRowMajor = 101 
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasConjNoTrans = 114

_libopenblas = libutils.get_ctypes_library('openblas', [])
if _libopenblas is None:
    HAVE_OPENBLAS = False
else:
    HAVE_OPENBLAS = True

class OpenblasCorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        super(OpenblasCorrProblem, self).__init__(size=size, dtype=dtype)
        if not HAVE_OPENBLAS:
            raise RuntimeError("Could not find libopenblas.so")
        # [C,Z]GBMV calculate the complex product of a banded matrix
        # with a vector.  In BLAS notation they compute:
        #      y <- alpha * A * x + beta * y
        # but in terms of our variables we compute:
        #      z <- 1.0 * conj(diag(x)) * y + 0.0 * z
        # since we need to preserve the inputs x and y, and the calling
        # sequence requires that we specify alpha and beta.  Note that
        # OpenBLAS specifies complex types as two-element arrays addressed
        # by reference.  To ensure the correct type of those constants,
        # we create appropriate length numpy arrays filled in with the
        # needed constants.

        if hasattr(_scheme.mgr.state, 'num_threads'):
            self.ncpus = _scheme.mgr.state.num_threads
        else:
            self.ncpus = 1

        _libopenblas.openblas_set_num_threads(self.ncpus)

        if dtype == complex64:
            self._efunc = _libopenblas.cblas_cgbmv
            self.alphavec = zeros(1, dtype = complex64)
            self.betavec = zeros(1, dtype = complex64)
        else:
            self._efunc = _libopenblas.cblas_zgbmv
            self.alphavec = zeros(1, dtype = complex128)
            self.betavec = zeros(1, dtype = complex128)

        self.alphavec[0] = 1.0 + 0.0j
        self.alpha = self.alphavec.ptr
        self.beta = self.betavec.ptr
        self.N = len(self.x)

        self._efunc.argtypes = [ctypes.c_int, ctypes.c_int, # ORDER, TRANSPOSE
                                ctypes.c_int, ctypes.c_int, # M, N
                                ctypes.c_int, ctypes.c_int, # KLOWER, KUPPER
                                ctypes.c_void_p, ctypes.c_void_p, # &ALPHA, &A = diag(x)
                                ctypes.c_int, ctypes.c_void_p, # LDA, &y
                                ctypes.c_int, ctypes.c_void_p, # incy, &beta
                                ctypes.c_void_p, ctypes.c_int] # &z, incz

        self.xptr = self.x.ptr
        self.yptr = self.y.ptr
        self.zptr = self.z.ptr

    def execute(self):
        self._efunc(CblasRowMajor, CblasConjNoTrans,
                    self.N, self.N, 0, 0,
                    self.alpha, self.xptr, 1, self.yptr, 1,
                    self.beta, self.zptr, 1)

    def _setup(self):
        pass

_libmkl = libutils.get_ctypes_library('mkl_rt', [])
if _libmkl is None:
    HAVE_MKL = False
else:
    HAVE_MKL = True

class MKLblasCorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        super(MKLblasCorrProblem, self).__init__(size=size, dtype=dtype)
        if not HAVE_MKL:
            raise RuntimeError("Could not find libmkl_rt.so")
        # [C,Z]GBMV calculate the complex product of a banded matrix
        # with a vector.  In BLAS notation they compute:
        #      y <- alpha * A * x + beta * y
        # but in terms of our variables we compute:
        #      z <- 1.0 * conj(diag(x)) * y + 0.0 * z
        # since we need to preserve the inputs x and y, and the calling
        # sequence requires that we specify alpha and beta.  Note that MKL
        # passes complex constants by reference.  To ensure the correct 
        # type of those constants, we create appropriate length numpy 
        # arrays filled in with the needed constants. Also note that MKL
        # does *NOT* support CblasConjNoTrans, so we use CblasConjTrans instead.

        if hasattr(_scheme.mgr.state, 'num_threads'):
            self.ncpus = _scheme.mgr.state.num_threads
        else:
            self.ncpus = 1

        if dtype == complex64:
            self._efunc = _libmkl.cblas_cgbmv
            self.alphavec = zeros(1, dtype = complex64)
            self.betavec = zeros(1, dtype = complex64)
        else:
            self._efunc = _libmkl.cblas_zgbmv
            self.alphavec = zeros(1, dtype = complex128)
            self.betavec = zeros(1, dtype = complex128)

        self.alphavec[0] = 1.0 + 0.0j
        self.alpha = self.alphavec.ptr
        self.beta = self.betavec.ptr
        self.N = len(self.x)

        self._efunc.argtypes = [ctypes.c_int, ctypes.c_int, # ORDER, TRANSPOSE
                                ctypes.c_int, ctypes.c_int, # M, N
                                ctypes.c_int, ctypes.c_int, # KLOWER, KUPPER
                                ctypes.c_void_p, ctypes.c_void_p, # &ALPHA, &A = diag(x)
                                ctypes.c_int, ctypes.c_void_p, # LDA, &y
                                ctypes.c_int, ctypes.c_void_p, # incy, &beta
                                ctypes.c_void_p, ctypes.c_int] # &z, incz

        self.xptr = self.x.ptr
        self.yptr = self.y.ptr
        self.zptr = self.z.ptr

    def execute(self):
        self._efunc(CblasRowMajor, CblasConjTrans,
                    self.N, self.N, 0, 0,
                    self.alpha, self.xptr, 1, self.yptr, 1,
                    self.beta, self.zptr, 1)

    def _setup(self):
        pass

simd_omp_code_float = """
int j;

#pragma omp parallel for schedule(static)
for (j = 0; j < NUM_THREADS; j++){
  ccmul(&aa[j*NBLOCK], &bb[j*NBLOCK], &cc[j*NBLOCK], NBLOCK);
}

"""

avx_support_save2 = """
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

//#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

static inline void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

int i;

#if _HAVE_AVX


// Unroll three times. Assume that one vector already conjugated.

for (i = 0; i < N-24; i += 24){
  __m256 a_re, a_im, b_flip;
  __m256 a1, b1, aib1, arb1;
  __m256 a2, b2, aib2, arb2;
  __m256 a3, b3, aib3, arb3;
 
  // Load everything into registers

  a1 = _mm256_load_ps(a+i);
  a2 = _mm256_load_ps(a+i+8);
  a3 = _mm256_load_ps(a+i+16);
  b1 = _mm256_load_ps(b+i);
  b2 = _mm256_load_ps(b+i+8);
  b3 = _mm256_load_ps(b+i+16);

  a_re = _mm256_shuffle_ps(a1, a1, 0xA0A0);
  arb1 = _mm256_mul_ps(a_re, b1);
  a_im = _mm256_shuffle_ps(a1, a1, 0xF5F5);
  b_flip = _mm256_shuffle_ps(b1, b1, 0xB1B1);
  aib1 = _mm256_mul_ps(a_im, b_flip);

  a_re = _mm256_shuffle_ps(a2, a2, 0xA0A0);
  arb2 = _mm256_mul_ps(a_re, b2);
  a_im = _mm256_shuffle_ps(a2, a2, 0xF5F5);
  b_flip = _mm256_shuffle_ps(b2, b2, 0xB1B1);
  aib2 = _mm256_mul_ps(a_im, b_flip);

  // Finish up first, reusing a1 register:
  a1 = _mm256_addsub_ps(arb1, aib1);

  a_re = _mm256_shuffle_ps(a3, a3, 0xA0A0);
  arb3 = _mm256_mul_ps(a_re, b3);
  a_im = _mm256_shuffle_ps(a3, a3, 0xF5F5);
  b_flip = _mm256_shuffle_ps(b3, b3, 0xB1B1);
  aib3 = _mm256_mul_ps(a_im, b_flip);

  a2 = _mm256_addsub_ps(arb2, aib2);
  a3 = _mm256_addsub_ps(arb3, aib3);

  _mm256_store_ps(c+i, a1);
  _mm256_store_ps(c+i+8, a2);
  _mm256_store_ps(c+i+16, a3);
}

#endif

}
"""

avx_support = """
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif


static inline void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

int i;

#if _HAVE_AVX

for (i = 0; i < N-8; i += 8){
  __m256 a_re, a_im, b_flip;
  __m256 a1, b1, aib1, arb1;
 
  // Load everything into registers

  a1 = _mm256_load_ps(a+i);
  b1 = _mm256_load_ps(b+i);

  a_re = _mm256_shuffle_ps(a1, a1, 0xA0A0);
  arb1 = _mm256_mul_ps(a_re, b1);
  a_im = _mm256_shuffle_ps(a1, a1, 0xF5F5);
  b_flip = _mm256_shuffle_ps(b1, b1, 0xB1B1);
  aib1 = _mm256_mul_ps(a_im, b_flip);
  a1 = _mm256_addsub_ps(arb1, aib1);

  _mm256_store_ps(c+i, a1);
}

#endif

}
"""


# In case we wanted to go back to this, but the different variations
# on AVX seem to make little difference, e.g. manual loop unrolling
# and reordering to hide instruction latency

avx_support_save = """
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

static inline void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

int i;

#if _HAVE_AVX

float minus_ones_vec[8] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
__m256 minus_ones;

minus_ones = _mm256_loadu_ps(minus_ones_vec);

// Unroll three times.  Interleave slightly differently to avoid need for
// multiple "minus_ones" registers.

for (i = 0; i < ROUND_DOWN(N, 24); i += 24){
  __m256 x1, y1, z1, areg1, breg1;
  __m256 x2, y2, z2, areg2, breg2;
  __m256 x3, y3, z3, areg3, breg3;

  // Load everything into registers

  areg1 = _mm256_load_ps(a+i);
  areg2 = _mm256_load_ps(a+i+8);
  areg3 = _mm256_load_ps(a+i+16);
  breg1 = _mm256_load_ps(b+i);
  breg2 = _mm256_load_ps(b+i+8);
  breg3 = _mm256_load_ps(b+i+16);

  x1 = _mm256_shuffle_ps(areg1, areg1, 0xA0A0);
  x2 = _mm256_shuffle_ps(areg2, areg2, 0xA0A0);
  x3 = _mm256_shuffle_ps(areg3, areg3, 0xA0A0);

  z1 = _mm256_mul_ps(x1, breg1);
  z2 = _mm256_mul_ps(x2, breg2);
  z3 = _mm256_mul_ps(x3, breg3);

  x1 = _mm256_shuffle_ps(areg1, areg1, 0xF5F5);
  x2 = _mm256_shuffle_ps(areg2, areg2, 0xF5F5);
  x3 = _mm256_shuffle_ps(areg3, areg3, 0xF5F5);

  x1 = _mm256_mul_ps(x1, minus_ones);
  y1 = _mm256_shuffle_ps(breg1, breg1, 0xB1B1);
  y1 = _mm256_mul_ps(x1, y1);
  x1 = _mm256_addsub_ps(z1, y1);

  x2 = _mm256_mul_ps(x2, minus_ones);
  y2 = _mm256_shuffle_ps(breg2, breg2, 0xB1B1);
  y2 = _mm256_mul_ps(x2, y2);
  x2 = _mm256_addsub_ps(z2, y2);

  x3 = _mm256_mul_ps(x3, minus_ones);
  y3 = _mm256_shuffle_ps(breg3, breg3, 0xB1B1);
  y3 = _mm256_mul_ps(x3, y3);
  x3 = _mm256_addsub_ps(z3, y3);

  _mm256_store_ps(c+i, x1);
  _mm256_store_ps(c+i+8, x2);
  _mm256_store_ps(c+i+16, x3);
}

#endif

}
"""

sse3_support = """
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#ifdef __SSE3__
#define _HAVE_SSE3 1
#else
#define _HAVE_SSE3 0
#endif

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

static inline void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

int i;

#if _HAVE_SSE3

// We load and store in reverse order, to avoid
// a multiplication by -1 implied by the conjugation
// of array 'a'.


// Unroll two times

for (i = 0; i < ROUND_DOWN(N, 4); i += 4){
  __m128 a1, b1, b1_flip, a1_im, a1_re, aib1, arb1, c1;
  __m128 a2, b2, b2_flip, a2_im, a2_re, aib2, arb2, c2;

  // Load everything into registers

  a1 = _mm_load_ps(a+i);
  a2 = _mm_load_ps(a+i+4);

  b1 = _mm_load_ps(b+i);
  b2 = _mm_load_ps(b+i+4);

  // Now calculations.  Note the following latencies:
  //     shuffle: 1 cycle
  //     addsub: 3 cycles
  //     mul: 5 cycles

  b1_flip = _mm_shuffle_ps(b1, b1, 0xB1); // Swap b1 re & im
  a1_im = _mm_shuffle_ps(a1, a1, 0xF5);   // Im(a1) in both
  aib1 = _mm_mul_ps(a1_im, b1_flip);      // (a1.im*b1.im, a1.im*b1.re)
  
  b2_flip = _mm_shuffle_ps(b2, b2, 0xB1); // Swap b2 re & im
  a2_im = _mm_shuffle_ps(a2, a2, 0xF5);   // Im(a2) in both
  aib2 = _mm_mul_ps(a2_im, b2_flip);      // (a2.im*b2.im, a2.im*b2.re)

  a1_re = _mm_shuffle_ps(a1, a1, 0xA0);   // Re(a1) in both
  arb1 = _mm_mul_ps(a1_re, b1);           // (a1.re*b1.re, a1.re*b1.im)

  a2_re = _mm_shuffle_ps(a2, a2, 0xA0);   // Re(a2) in both
  arb2 = _mm_mul_ps(a2_re, b2);           // (a1.re*b1.re, a1.re*b1.im)

  c1 = _mm_addsub_ps(arb1, aib1);           // subtract/add
  c2 = _mm_addsub_ps(arb2, aib2);           // subtract/add

  _mm_store_ps(c+i, c1);
  _mm_store_ps(c+i+4, c2);

}

#endif

}
"""



# Patterned after the blaze library, but including conjugation
# of first vector.
simd_code_float = """
ccmul(aa, bb, cc, NN);
"""
class SIMDCorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        super(SIMDCorrProblem, self).__init__(size=size, dtype=dtype)
        if dtype != complex64:
            raise RuntimeError("SIMD only supports single-precision complex.")
        self.thecode = simd_code_float

    def execute(self):
        aa = _np.array(self.x.data, copy=False).view(dtype = float32)
        bb = _np.array(self.y.data, copy=False).view(dtype = float32)
        cc = _np.array(self.z.data, copy=False).view(dtype = float32)
        NN = 2 * len(self.x)        
        inline(self.thecode, ['aa', 'bb', 'cc', 'NN'],
               extra_compile_args=['-march=native  -O3 -w'],
#               extra_compile_args=['-msse3 -O3 -w'],
               support_code = avx_support,
               auto_downcast = 1)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

### ASOA

simd_asoa_support = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

//#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

static inline void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

int i;

#if _HAVE_AVX

/*

We only have enough registers to unroll the loop twice.

*/


__m256 ar1, ai1, br1, bi1, arbr1, aibi1, arbi1, aibr1;
__m256 ar2, ai2, br2, bi2, arbr2, aibi2, arbi2, aibr2;

for (i = 0; i < N-32; i += 32){

  // Load up all of the registers
  ar1 = _mm256_load_ps(a+i); 
  ai1 = _mm256_load_ps(a+i+8);
  ar2 = _mm256_load_ps(a+i+16); 
  ai2 = _mm256_load_ps(a+i+24);
  br1 = _mm256_load_ps(b+i);
  bi1 = _mm256_load_ps(b+i+8);
  br2 = _mm256_load_ps(b+i+16);
  bi2 = _mm256_load_ps(b+i+24);

  arbr1 = _mm256_mul_ps(ar1, br1);
  aibi1 = _mm256_mul_ps(ai1, ai1);
  arbi1 = _mm256_mul_ps(ar1, ai1);
  aibr1 = _mm256_mul_ps(ai1, ar1);

  arbr2 = _mm256_mul_ps(ar2, br2);
  aibi2 = _mm256_mul_ps(ai2, ai2);

  // Re-use registers for output
  ar1 = _mm256_add_ps(arbr1, aibi1);

  // Return to computing second array element

  arbi2 = _mm256_mul_ps(ar2, ai2);
  aibr2 = _mm256_mul_ps(ai2, ar2);

  // Compute second output for first
  ai1 = _mm256_sub_ps(arbi1, aibr1);

  // Now we can't hide instruction latency anymore
  ar2 = _mm256_add_ps(arbr2, aibi2);
  ai2 = _mm256_sub_ps(arbi2, aibr2);

  // Store results

  _mm256_store_ps(c+i, ar1);
  _mm256_store_ps(c+i+8, ai1);
  _mm256_store_ps(c+i+16, ar2);
  _mm256_store_ps(c+i+24, ai2);

  }
}

#else
#error AVX not available
#endif

"""

simd_asoa_code = """
int j;

#pragma omp parallel for schedule(static)
for (j = 0; j < NUM_THREADS; j++){
  ccmul(&aa[j*NBLOCK], &bb[j*NBLOCK], &cc[j*NBLOCK], NBLOCK);
}
"""

simd_asoa_support_save = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

//#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

static inline void ccmul(float * __restrict are, float * __restrict aim,
                         float * __restrict bre, float * __restrict bim,
                         float * __restrict cre, float * __restrict cim,
                         int N){

int i;

#if _HAVE_AVX

/*

We only have enough registers to unroll the loop twice.

*/


__m256 ar1, ai1, br1, bi1, arbr1, aibi1, arbi1, aibr1;
//__m256 ar2, ai2, br2, bi2, arbr2, aibi2, arbi2, aibr2;

for (i = 0; i < N-8; i += 8){

  // Load up all of the registers
  ar1 = _mm256_load_ps(are+i); 
  //ar2 = _mm256_load_ps(are+i+8); 
  ai1 = _mm256_load_ps(aim+i);
  //ai2 = _mm256_load_ps(aim+i+8);
  br1 = _mm256_load_ps(bre+i);
  //br2 = _mm256_load_ps(bre+i+8);
  bi1 = _mm256_load_ps(bim+i);
  //bi2 = _mm256_load_ps(bim+i+8);

  arbr1 = _mm256_mul_ps(ar1, br1);
  aibi1 = _mm256_mul_ps(ai1, ai1);
  arbi1 = _mm256_mul_ps(ar1, ai1);
  aibr1 = _mm256_mul_ps(ai1, ar1);

  //arbr2 = _mm256_mul_ps(ar2, br2);
  //aibi2 = _mm256_mul_ps(ai2, ai2);

  // Re-use registers for output
  ar1 = _mm256_add_ps(arbr1, aibi1);

  // Return to computing second array element

  //arbi2 = _mm256_mul_ps(ar2, ai2);
  //aibr2 = _mm256_mul_ps(ai2, ar2);

  // Compute second output for first
  ai1 = _mm256_sub_ps(arbi1, aibr1);

  // Now we can't hide instruction latency anymore
  //ar2 = _mm256_add_ps(arbr2, aibi2);
  //ai2 = _mm256_sub_ps(arbi2, aibr2);

  // Store results

  _mm256_store_ps(cre+i, ar1);
  //_mm256_store_ps(cre+i+8, ar2);
  _mm256_store_ps(cim+i, ai1);
  //_mm256_store_ps(cim+i+8, ai2);

  }
}

#else
#error AVX not available
#endif

"""

simd_asoa_code_save = """
int j;

#pragma omp parallel for schedule(static)
for (j = 0; j < NUM_THREADS; j++){
  ccmul(&aa_re[j*NBLOCK], &aa_im[j*NBLOCK],
        &bb_re[j*NBLOCK], &bb_im[j*NBLOCK],
        &cc_re[j*NBLOCK], &cc_im[j*NBLOCK],
        NBLOCK);
}
"""

class SIMD_ASOA_CorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        super(SIMD_ASOA_CorrProblem, self).__init__(size=size, dtype=dtype)
        if dtype != complex64:
            raise RuntimeError("SIMD only supports single-precision complex.")
        n = size/max_chunk
        tmpcode = simd_asoa_code.replace('NUM_THREADS', str(n))
        self.thecode = tmpcode.replace('NBLOCK', str(max_chunk))

    def execute(self):
        aa = _np.array(self.x.data, copy=False).view(dtype = float32)
        bb = _np.array(self.y.data, copy=False).view(dtype = float32)
        cc = _np.array(self.z.data, copy=False).view(dtype = float32)
        inline(self.thecode, ['aa', 'bb', 'cc'],
               extra_compile_args=['-march=native -fprefetch-loop-arrays -O3 -w'] + omp_flags,
               support_code = simd_asoa_support,
               auto_downcast = 1,
               libraries = omp_libs)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


class AVX_OMP_CorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        if dtype != complex64:
            raise RuntimeError("SIMD only supports single-precision complex.")
        super(AVX_OMP_CorrProblem, self).__init__(size=size, dtype=dtype)
        n = size/max_chunk
        tmpcode = simd_omp_code_float.replace('NUM_THREADS', str(n))
        self.thecode = tmpcode.replace('NBLOCK', str(max_chunk))

    def execute(self):
        aa = _np.array(self.x.data, copy=False).view(dtype = float32)
        bb = _np.array(self.y.data, copy=False).view(dtype = float32)
        cc = _np.array(self.z.data, copy=False).view(dtype = float32)
        inline(self.thecode, ['aa', 'bb', 'cc'],
               extra_compile_args=['-march=native -fprefetch-loop-arrays -O3 -w'] + omp_flags,
               #extra_compile_args=['-msse3 -O3 -w'] + omp_flags,
               support_code = avx_support,
               auto_downcast = 1,
               libraries = omp_libs)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


class SSE3_OMP_CorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        if dtype != complex64:
            raise RuntimeError("SIMD only supports single-precision complex.")
        super(SSE3_OMP_CorrProblem, self).__init__(size=size, dtype=dtype)
        n = size/max_chunk
        tmpcode = simd_omp_code_float.replace('NUM_THREADS', str(n))
        self.thecode = tmpcode.replace('NBLOCK', str(max_chunk))

    def execute(self):
        aa = _np.array(self.x.data, copy=False).view(dtype = float32)
        bb = _np.array(self.y.data, copy=False).view(dtype = float32)
        cc = _np.array(self.z.data, copy=False).view(dtype = float32)
        inline(self.thecode, ['aa', 'bb', 'cc'],
               extra_compile_args=['-march=native -fprefetch-loop-arrays -O3 -w'] + omp_flags,
               #extra_compile_args=['-msse3 -O3 -w'] + omp_flags,
               support_code = sse3_support,
               auto_downcast = 1,
               libraries = omp_libs)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


static_omp_support = """
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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

class Static_OMP_CorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        if dtype != complex64:
            raise RuntimeError("SIMD only supports single-precision complex.")
        super(Static_OMP_CorrProblem, self).__init__(size=size, dtype=dtype)
        n = size/max_chunk
        tmpcode = simd_omp_code_float.replace('NUM_THREADS', str(n))
        self.thecode = tmpcode.replace('NBLOCK', str(max_chunk))

    def execute(self):
        aa = _np.array(self.x.data, copy=False).view(dtype = float32)
        bb = _np.array(self.y.data, copy=False).view(dtype = float32)
        cc = _np.array(self.z.data, copy=False).view(dtype = float32)
        inline(self.thecode, ['aa', 'bb', 'cc'],
               #extra_compile_args=['-march=native -fprefetch-loop-arrays -funroll-loops -O3 -w'] + omp_flags,
               extra_compile_args=['-O3 -w'] + omp_flags,
               support_code = static_omp_support,
               auto_downcast = 1,
               libraries = omp_libs)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


_dtype_dict = { 's' : complex64,
                'd' : complex128 }

_class_dict = { 'numpy' : NumpyCorrProblem,
                'weave' : WeaveCorrProblem,
                'openblas' : OpenblasCorrProblem,
                'mkl' : MKLblasCorrProblem,
                'simd' : SIMDCorrProblem,
                'simd_asoa' : SIMD_ASOA_CorrProblem,
                'avx_omp' : AVX_OMP_CorrProblem,
                'sse3_omp' : SSE3_OMP_CorrProblem,
                'static_omp' : Static_OMP_CorrProblem
                }

corr_valid_methods = _class_dict.keys()

def parse_corr_problem(probstring, method='numpy'):
    """
    This function takes a string of the form [s|d]<number>, where:
       [s|d] is 'single' or 'double'; the precision of the arrays

    It also takes another argument, a string indicating which class
    type to return. 

    It returns the class, size, and dtype, so that the call:
        MyClass, n, dt = parse_corr_problem(probstring, method)
    should usually be followed by:
        MyProblem = MyClass(n, dtype = dt)
    """
    dtype = _dtype_dict[probstring[0:1]]
    prob_class = _class_dict[method]
    size = int(probstring[1:])

    return prob_class, size, dtype
