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
        super(NewPyCBCCorrProblem, self).__init__(size=size, dtype=dtype)

    def execute(self):
        correlate_simd(self.x, self.y, self.z)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


simd_omp_code_float = """
int j;

#pragma omp parallel for schedule(static)
for (j = 0; j < NUM_THREADS; j++){
  ccmul(&aa[j*NBLOCK], &bb[j*NBLOCK], &cc[j*NBLOCK], NBLOCK);
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
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
  __m256 ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
  float *aptr, *bptr, *cptr;

  aptr = a;
  bptr = b;
  cptr = c;

  for (i = 0; i < N; i += 32){
      // Load everything into registers

      //ymm0 = _mm256_load_ps(aptr);
      //ymm3 = _mm256_load_ps(aptr+8);
      //ymm6 = _mm256_load_ps(aptr+16);
      //ymm9 = _mm256_load_ps(aptr+24);
      _mm256_stream_si256((__m256i *) aptr, _mm256_castps_si256(ymm0));
      _mm256_stream_si256((__m256i *) (aptr+8), _mm256_castps_si256(ymm3));
      _mm256_stream_si256((__m256i *) (aptr+16), _mm256_castps_si256(ymm6));
      _mm256_stream_si256((__m256i *) (aptr+24), _mm256_castps_si256(ymm9));
      //ymm1 = _mm256_load_ps(bptr);
      //ymm4 = _mm256_load_ps(bptr+8);
      //ymm7 = _mm256_load_ps(bptr+16);
      //ymm10 = _mm256_load_ps(bptr+24);
      _mm256_stream_si256((__m256i *) bptr, _mm256_castps_si256(ymm1));
      _mm256_stream_si256((__m256i *) (bptr+8), _mm256_castps_si256(ymm4));
      _mm256_stream_si256((__m256i *) (bptr+16), _mm256_castps_si256(ymm7));
      _mm256_stream_si256((__m256i *) (bptr+24), _mm256_castps_si256(ymm10));

      ymm2 = _mm256_movehdup_ps(ymm1);
      ymm1 = _mm256_moveldup_ps(ymm1);
      ymm1 = _mm256_mul_ps(ymm1, ymm0);
      ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
      ymm2 = _mm256_mul_ps(ymm2, ymm0);
      ymm0 = _mm256_addsub_ps(ymm1, ymm2);

      ymm5 = _mm256_movehdup_ps(ymm4);
      ymm4 = _mm256_moveldup_ps(ymm4);
      ymm4 = _mm256_mul_ps(ymm4, ymm3);
      ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
      ymm5 = _mm256_mul_ps(ymm5, ymm3);
      ymm3 = _mm256_addsub_ps(ymm4, ymm5);

      ymm8 = _mm256_movehdup_ps(ymm7);
      ymm7 = _mm256_moveldup_ps(ymm7);
      ymm7 = _mm256_mul_ps(ymm7, ymm6);
      ymm6 = _mm256_shuffle_ps(ymm6, ymm6, 0xB1);
      ymm8 = _mm256_mul_ps(ymm8, ymm6);
      ymm6 = _mm256_addsub_ps(ymm7, ymm8);

      ymm11 = _mm256_movehdup_ps(ymm10);
      ymm10 = _mm256_moveldup_ps(ymm10);
      ymm10 = _mm256_mul_ps(ymm10, ymm9);
      ymm9 = _mm256_shuffle_ps(ymm9, ymm9, 0xB1);
      ymm11 = _mm256_mul_ps(ymm11, ymm9);
      ymm9 = _mm256_addsub_ps(ymm10, ymm11);

      _mm256_store_ps(cptr, ymm0);
      _mm256_store_ps(cptr+8, ymm3);
      _mm256_store_ps(cptr+16, ymm6);
      _mm256_store_ps(cptr+24, ymm9);

      aptr += 32;
      bptr += 32;
      cptr += 32;
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


class AVX_OMP_CorrProblem(BaseCorrProblem):
    def __init__(self, size, dtype):
        if dtype != complex64:
            raise RuntimeError("SIMD only supports single-precision complex.")
        super(AVX_OMP_CorrProblem, self).__init__(size=size, dtype=dtype)
        n = 2*size/max_chunk
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
        n = 2*size/max_chunk
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
        n = 2*size/max_chunk
        tmpcode = simd_omp_code_float.replace('NUM_THREADS', str(n))
        self.thecode = tmpcode.replace('NBLOCK', str(max_chunk))

    def execute(self):
        aa = _np.array(self.x.data, copy=False).view(dtype = float32)
        bb = _np.array(self.y.data, copy=False).view(dtype = float32)
        cc = _np.array(self.z.data, copy=False).view(dtype = float32)
        inline(self.thecode, ['aa', 'bb', 'cc'],
               extra_compile_args=['-march=native -ffast-math -fprefetch-loop-arrays -funroll-loops -O3 -w'] + omp_flags,
               #extra_compile_args=['-O3 -w'] + omp_flags,
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
                'avx_omp' : AVX_OMP_CorrProblem,
                'sse3_omp' : SSE3_OMP_CorrProblem,
                'static_omp' : Static_OMP_CorrProblem,
                'new_pycbc' : NewPyCBCCorrProblem
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
