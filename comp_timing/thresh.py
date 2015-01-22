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
from pycbc.events.threshold_cpu import threshold_inline, threshold_numpy
try:
    from pycbc.events.threshold_cpu import threshold_simd
except:
    threshold_simd = None
from pycbc.events.threshold_cpu import omp_flags, omp_libs
from pycbc import scheme as _scheme
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils
from math import sqrt

class BaseThreshProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        # We'll do some arithmetic with these, so sanity check first:
        if (size < 1):
            raise ValueError("size must be >= 1")

        self.input = zeros(size, dtype=complex64)
        self.vals = zeros(size, dtype=complex64)
        self.locs = _np.zeros(size, dtype=_np.uint32)
        self.count = _np.zeros(1, dtype=_np.uint32)
        self.threshhold = _np.zeros(1, dtype=float32)

        self.threshhold[0] = 1.0

#        fac1 = int(sqrt(size))
#        fac2 = int(size/fac1)
        fac1 = int(size/16)
        fac2 = 16
        for i in range(0,fac1):
            self.input[fac2*i] = 1.1*self.threshhold[0]

# Several of the OpenMP based approaches use this
#max_chunk = 4096
max_chunk = 8192
#max_chunk = 131072
#max_chunk = 65536
#max_chunk = 32768

# Now our derived classes
class NumpyThreshProblem(BaseThreshProblem):
    def __init__(self, size):
        super(NumpyThreshProblem, self).__init__(size=size)
            
    def execute(self):
        self.locs, self.vals = threshold_numpy(self.input, self.threshhold[0])

    def _setup(self):
        pass

class WeaveThreshProblem(BaseThreshProblem):
    def __init__(self, size):
        super(WeaveThreshProblem, self).__init__(size=size)

    def execute(self):
        threshold_inline(self.input, self.threshhold[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

class NewPyCBCThreshProblem(BaseThreshProblem):
    def __init__(self, size):
        if threshold_simd is None:
            raise RuntimeError("Your version of PyCBC does not have threshold_simd")
        super(NewPyCBCThreshProblem, self).__init__(size=size)

    def execute(self):
        threshold_simd(self.input, self.threshhold[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()



new_support_save = """
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <x86intrin.h>
#include <stdint.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

void count_thresh(const float* __restrict__ input, 
                  unsigned int* __restrict__ count_loc, const float tsqr, 
                  const unsigned int N){

  // This function passes through the data first, counting the number
  // of points in each parallel segment that are above threshhold.
  //
  // Note that we pass in the squared value of the threshhold.

  unsigned int i, c;

  c = 0;

#if _HAVE_AVX

  __m256 thr2, in2, in2_swap;
  //__m128i c_lo, c_hi, inc_lo, inc_hi;
  //int64_t sums[4] __attribute__ ((aligned(32)));
  int mask;
  // Put it on a cache line boundary
  const unsigned int count_table[16] __attribute__ ((aligned(64))) = {0, 1, 1, 2,
                                                                      1, 2, 2, 3,
                                                                      1, 2, 2, 3,
                                                                      2, 3, 3, 4};                                        
  _mm_prefetch(count_table, _MM_HINT_T0);

  //_mm256_zeroall(); // Note this sets c_lo and c_hi to zero
  thr2 = _mm256_broadcast_ss(&tsqr);
  //c_lo = _mm_setzero_si128();
  //c_hi = _mm_setzero_si128();

  for (i = 0; i < N-8; i += 8){
    in2 = _mm256_load_ps(input+i);
    in2 = _mm256_mul_ps(in2, in2);                  // re*re, im*im
    in2_swap = _mm256_shuffle_ps(in2, in2, 0xB1B1); // swap real and imaginary
    in2 = _mm256_add_ps(in2, in2_swap);             // Add, so now re^2 +im^2
    in2 = _mm256_cmp_ps(in2, thr2, _CMP_GT_OQ);     // Compare to squared threshold:
    // The next steps because packed integer addition requires AVX2 (Haswell)
    //inc_lo = _mm256_extractf128_si256(_mm256_castps_si256(in2), 0);
    //inc_hi = _mm256_extractf128_si256(_mm256_castps_si256(in2), 1);
    //_mm256_zeroupper(); // To avoid AVX/SSE transition penalty
    //c_lo = _mm_add_epi64(c_lo, inc_lo);
    //c_hi = _mm_add_epi64(c_hi, inc_hi);
    // Extract high bits of comparison into int. Cast allows 16 entry table
    // rather than 256 entry table
    mask = _mm256_movemask_pd(_mm256_castps_pd(in2)); 
    c += count_table[mask];
  }
  
  //_mm_store_si128((__m128i *) &sums[0], c_lo);
  //_mm_store_si128((__m128i *) &sums[2], c_hi);
  //c = (unsigned int)(-sums[0]-sums[1]-sums[2]-sums[3]);

#elif _HAVE_SSE3

  __m128 thr2, in2, in2_swap;
  int mask;
  unsigned int count_table[4] = {0, 1, 1, 2};

  thr2 = _mm_broadcast_ss(&tsqr);

  for (i = 0; i < N-4; i += 4){
    in2 = _mm_load_ps(input+i);
    in2 = _mm_mul_ps(in2, in2);                  // re*re, im*im
    in2_swap = _mm_shuffle_ps(in2, in2, 0xB1);   // swap real and imaginary
    in2 = _mm_add_ps(in2, in2_swap);             // Add, so now re^2 +im^2
    in2 = _mm_cmpgt_ps(in2, thr2);               // Compare to squared threshold:
    // Extract high bits of comparison into int. Cast allows 4 entry table
    // rather than 16 entry table
    mask = _mm_movemask_pd(_mm_castps_pd(in2)); 
    c += count_table[mask];
  }

#else

  float re, im;

  for (i = 0; i < N; i +=2){
    re = input[i];
    im = input[i+1];
    if ( (re*re + im*im) > tsqr){
      c++;
    }
  }

#endif

  *count_loc = c;

  return;
}

static inline unsigned int excl_prefix_sum(unsigned int* __restrict__ count_arr, const unsigned int len){

  unsigned int i, inim1, tmp;

  // This function does a (serial) exclusive prefix sum on the array count_arr, in-place.
  // It returns the sum of *all* elements in the array (which would conceptually be the
  // value that would be just past the end of the exclusive sum array). 

  inim1 = count_arr[0];
  count_arr[0] = 0;
  for (i = 1; i < len; i++){
    tmp = count_arr[i];
    count_arr[i] = inim1 + count_arr[i-1];
    inim1 = tmp;
  }

  return (count_arr[len-1] + inim1);

}

static inline void copy_above(const float* __restrict__ input, 
                              float* __restrict__ vals, unsigned int * __restrict__ locs, 
                              const float tsqr, const unsigned int N,
                              const unsigned int in_offset) {

 // This function passes through the data once more, using the number of points above threshhold
 // in this segment to know (deterministically) where to write the values and locations of those
 // points that are above threshhold.  It is *critical* to the correct behavior of the overall
 // algorithm that the segmentation and also the matching between a segment and threadno be the same
 // for this function as for the preceding two.


#if _HAVE_AVX

  __m128 thr2, in2, in2_swap;
  __m128d inval, curr, next, inrev;
  __m128i mask;
  int above, valid;
  unsigned int i, c;
  // Put each LUT  on a cache line boundary
  // We use 32-bit *signed* integers; in twos-complement repr, -1 is all bits = 1
  const int32_t rev_table[32] __attribute__ ((aligned(64))) = {0, 0, 0, 0,     // 000
                                                               -1, -1, 0, 0,   // 001
                                                               0, 0, 0, 0,     // 010
                                                               -1, -1, -1, -1, // 011
                                                               -1, -1, 0, 0,   // 100
                                                               -1, -1, 0, 0,   // 101
                                                               -1, -1, -1, -1, // 110
                                                               -1, -1, 0, 0};  // 111

  const int32_t val_table[32] __attribute__ ((aligned(64))) = {0, 0, 0, 0,    // 000
                                                               -1, -1, 0, 0,  // 001
                                                               -1, -1, 0, 0,  // 010
                                                               0, 0, 0, 0,    // 011
                                                               -1, -1, 0, 0,  // 100
                                                               0, 0, 0, 0,    // 101
                                                               0, 0, 0, 0,    // 110
                                                               -1, -1, 0, 0}; // 111

  const int32_t wri_table[32] __attribute__ ((aligned(64))) = {0, 0, 0, 0,      // 000
                                                               0, 0, 0, 0,      // 001
                                                               0, 0, 0, 0,      // 010
                                                               -1, -1, -1, -1,  // 011
                                                               0, 0, 0, 0,      // 100
                                                               -1, -1, -1, -1,  // 101
                                                               -1, -1, -1, -1,  // 110
                                                               -1, -1, -1, -1}; // 111

  // Preload all the lookup tables:
  _mm_prefetch(rev_table, _MM_HINT_T0);
  _mm_prefetch(val_table, _MM_HINT_T0);
  _mm_prefetch(wri_table, _MM_HINT_T0);

  // Set initial values
  valid = 0;
  c = 0;
  thr2 = _mm_broadcast_ss(&tsqr);
  curr = _mm_setzero_pd();
  next = _mm_setzero_pd();
  mask = _mm_setzero_si128();

  for (i = 0; i < N-4; i += 4){
    inval = _mm_load_pd((double *) &input[i]);
    in2 = _mm_mul_ps(_mm_castpd_ps(inval), _mm_castpd_ps(inval)); // re*re, im*im
    in2_swap = _mm_shuffle_ps(in2, in2, 0xB1);   // swap real and imaginary
    in2 = _mm_add_ps(in2, in2_swap);             // Add, so now re^2 +im^2
    in2 = _mm_cmpgt_ps(in2, thr2);               // Compare to squared threshold:

    // Extract high bits of comparison into int. Cast allows 4 entry table
    // rather than 16 entry table
    above = _mm_movemask_pd(_mm_castps_pd(in2)); // So 'above' is 00, 01, 10 or 11
    above += (valid << 2); // Now above is 000---111

    // Our various blend and write operations work with masks.  We'd like to
    // avoid lookups and work with fast operations, if we can.  Because there is
    // no variable permute in SSE (only AVX and later) we create the reverse of
    // the input unconditionally, and blend both the original order and 
    // reverse into both current and next:
    //    curr = blendv(orig, curr, m1)
    //    curr = blendv(rev, curr,  m2)
    //    next = blendv(orig, rev,  m3)
    //    write(curr, m4)
    //    curr = blendv(next, curr, ~m4)
    // Note that the mask for overwriting curr with next must always be the negation
    // of the mask for writing out curr, since we don't want to overwrite if we haven't
    // written out; if we have, it can never hurt to do so.

    // Below, we show curr[0] > 0 , curr[1] > 0, inval[0] > thr, inval[1] > thr,
    // and the corresponding masks m1--m4.  Note that having both curr[0,1] > thr
    // is impossible, since in that case we would have written out. The blendv
    // instruction writes out the second argument when the corresponding bit is
    // set, otherwise the first; for this reason we give 'curr' as the second
    // argument, since then in all cases where we blend with curr, if its lowest
    // complex is valid, then the bit should be set, otherwise not.
    //
    // c1c0i1i0        m1     m2      m3     m4   New cur_valid
    // ======================================================================
    // 0000            **     **      **     00      00            00
    // 0001            *0     *1      **     00      01            10  
    // 0010            **     *0      **     00      01            01
    // 0011            00     11      **     11      00            11
    // 0100            *1     *1      **     00      01            00
    // 0101            *1     01      **     11      00            10
    // 0110            01     11      **     11      00            01
    // 0111            *1     01      *1     11      01            11
    //
    // Examining this, we see that:
    //   m1_lo = cur_valid_lo, m1_hi = zeros  ==> m1 = cur_valid

    inrev = _mm_shuffle_pd(inval, inval, 0x2);   // swap first and second
    curr = _mm_blendv_pd(inval, curr, _mm_castsi128_pd(mask)); // Note mask was set last time through
    mask = _mm_load_si128((__m128i *) &rev_table[4*above]);
    curr = _mm_blendv_pd(inrev, curr, _mm_castsi128_pd(mask));
    next = _mm_blendv_pd(inval, inrev, _mm_castsi128_pd(mask));
    mask = _mm_load_si128((__m128i *) &wri_table[4*above]);
    _mm_maskmoveu_si128(_mm_castpd_si128(curr), mask, (char *) &vals[c]);
    //_mm256_zeroupper();
    //_mm_maskstore_pd((double *) &vals[c], mask, curr);
    //_mm256_zeroupper();
    c += 2*(_mm_movemask_pd(_mm_castsi128_pd(mask)))-2;
    mask = _mm_load_si128((__m128i *) &val_table[4*above]);
    valid = _mm_movemask_pd(_mm_castsi128_pd(mask));
  }

#else

  unsigned int offset, i, c;
  float re, im, t2;

  // Copies:
  t2 = tsqr;

  c = 0;
  for (i = 0; i < N; i += 2){
    re = input[i];
    im = input[i+1];
    if ((re*re + im*im) > t2){
      vals[c] = re;
      vals[c + 1] = im;
      locs[c/2] = i/2 + in_offset;
      c += 2;
    }
  }


#endif


}
"""

new_support = """
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <x86intrin.h>
#include <stdint.h>

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

#ifdef __SSE3__
#define _HAVE_SSE3 1
#else
#define _HAVE_SSE3 0
#endif

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

void count_thresh(const float* __restrict__ input, 
                  unsigned int* __restrict__ count_loc, const float tsqr, 
                  const unsigned int N){

  // This function passes through the data first, counting the number
  // of points in each parallel segment that are above threshhold.
  //
  // Note that we pass in the squared value of the threshhold.

  unsigned int i, c;

  c = 0;

#if _HAVE_AVX

  __m256 thr2, in2, in2_swap;
  int mask;
  // Put it on a cache line boundary
  const unsigned int count_table[16] __attribute__ ((aligned(64))) = {0, 1, 1, 2,
                                                                      1, 2, 2, 3,
                                                                      1, 2, 2, 3,
                                                                      2, 3, 3, 4};                                        
  _mm_prefetch(count_table, _MM_HINT_T0);

  thr2 = _mm256_broadcast_ss(&tsqr);

  for (i = 0; i < N; i += 8){
    in2 = _mm256_load_ps(input+i);
    in2 = _mm256_mul_ps(in2, in2);                  // re*re, im*im
    in2_swap = _mm256_shuffle_ps(in2, in2, 0xB1B1); // swap real and imaginary
    in2 = _mm256_add_ps(in2, in2_swap);             // Add, so now re^2 +im^2
    in2 = _mm256_cmp_ps(in2, thr2, _CMP_GT_OQ);     // Compare to squared threshold:
    // Extract high bits of comparison into int. Cast allows 16 entry table
    // rather than 256 entry table
    mask = _mm256_movemask_pd(_mm256_castps_pd(in2)); 
    c += count_table[mask];
  }
  
#elif _HAVE_SSE3

  __m128 thr2, in2, in2_swap;
  int mask;
  unsigned int count_table[4] = {0, 1, 1, 2};

  thr2 = _mm_broadcast_ss(&tsqr);

  for (i = 0; i < N; i += 4){
    in2 = _mm_load_ps(input+i);
    in2 = _mm_mul_ps(in2, in2);                  // re*re, im*im
    in2_swap = _mm_shuffle_ps(in2, in2, 0xB1);   // swap real and imaginary
    in2 = _mm_add_ps(in2, in2_swap);             // Add, so now re^2 +im^2
    in2 = _mm_cmpgt_ps(in2, thr2);               // Compare to squared threshold:
    // Extract high bits of comparison into int. Cast allows 4 entry table
    // rather than 16 entry table
    mask = _mm_movemask_pd(_mm_castps_pd(in2)); 
    c += count_table[mask];
  }

#else

  float re, im;

  for (i = 0; i < N; i +=2){
    re = input[i];
    im = input[i+1];
    if ( (re*re + im*im) > tsqr){
      c++;
    }
  }

#endif

  *count_loc = c;

  return;
}

static inline unsigned int excl_prefix_sum(unsigned int* __restrict__ count_arr, const unsigned int len){

  unsigned int i, inim1, tmp;

  // This function does a (serial) exclusive prefix sum on the array count_arr, in-place.
  // It returns the sum of *all* elements in the array (which would conceptually be the
  // value that would be just past the end of the exclusive sum array). 

  inim1 = count_arr[0];
  count_arr[0] = 0;
  for (i = 1; i < len; i++){
    tmp = count_arr[i];
    count_arr[i] = inim1 + count_arr[i-1];
    inim1 = tmp;
  }

  return (count_arr[len-1] + inim1);

}

static inline void copy_above(const float* __restrict__ input, 
                              float* __restrict__ vals, unsigned int * __restrict__ locs, 
                              const float tsqr, const unsigned int N,
                              const unsigned int in_offset) {

 // This function passes through the data once more, using the number of points above threshhold
 // in this segment to know (deterministically) where to write the values and locations of those
 // points that are above threshhold.  It is *critical* to the correct behavior of the overall
 // algorithm that the segmentation and also the matching between a segment and threadno be the same
 // for this function as for the preceding two.

  unsigned int offset, i, c;
  float re, im, t2;

  // Copies:
  t2 = tsqr;

  c = 0;
  for (i = 0; i < N; i += 2){
    re = input[i];
    im = input[i+1];
    if ((re*re + im*im) > t2){
      vals[c] = re;
      vals[c + 1] = im;
      locs[c/2] = i/2 + in_offset;
      c += 2;
    }
  }

}
"""

new_code = """
// This code performs an OMP parallelized threshholding using the
// three functions defined in 'new_support' above. The basic strategy
// is to make a pass over the input, in appropriate sized chunks, first
// counting in each chunk the number of values above threshhold and
// storing that in an array; this is parallelized.  Next follows a
// serialized exclusive prefix sum of the array of above-threshhold
// counts.  Finally, the data is read again, and every value above
// threshhold is written out (along with its index) to the appropriate
// array. This can again be parallelized, as we now know deterministically
// where each output in a chunk should be written in the final arrays.
//
// The OMP coding is somewhat subtle.  The essential difficulty is that
// the 'chunking' and association of counts to the count_array must
// remain the same between for the invocations of all three functions.

// Note that NCHUNKS, CHUNKSIZE, and NLEN must all be calculated before
// weave compilation of this code fragment, and the appropriate string
// substitutions made.

unsigned int chunk_counts[NCHUNKS] __attribute__(( aligned(32) ));
unsigned int i;
float tsqr;

tsqr = threshhold * threshhold;

#pragma omp parallel for schedule(static, 1)
for (i = 0; i < NCHUNKS; i++){
  count_thresh(&inarr[CHUNKSIZE*i], &chunk_counts[i], tsqr, CHUNKSIZE);  
}

count[0] = excl_prefix_sum(chunk_counts, NCHUNKS);

#pragma omp parallel for schedule(static, 1)
for (i = 0; i < NCHUNKS; i++){
  copy_above(&inarr[CHUNKSIZE*i], &vals[2*chunk_counts[i]], &locs[chunk_counts[i]], tsqr, CHUNKSIZE, CHUNKSIZE*i); 
}
"""

class StaticThreshProblem(BaseThreshProblem):
    def __init__(self, size):
        super(StaticThreshProblem, self).__init__(size=size)
        if hasattr(_scheme.mgr.state, 'num_threads'):
            ncpus = _scheme.mgr.state.num_threads
        else:
            ncpus = 1
        work_by_cores = int(size/ncpus)
        self.chunksize = max_chunk

        print "size = {0}, ncpus = {1}, CHUNKSIZE = {2}".format(size, ncpus, self.chunksize)
        self.thecode = new_code.replace('CHUNKSIZE', str(self.chunksize))
        self.thecode = self.thecode.replace('NLEN', str(size))
        nchunk = int(2*size/self.chunksize)
        if (nchunk*self.chunksize != size):
            # We didn't divide evenly
            self.nchunk = nchunk + 1
        else:
            self.nchunk = nchunk
        self.thecode = self.thecode.replace('NCHUNKS', str(self.nchunk))

    def execute(self):
        inarr = _np.array(self.input.data, copy=False).view(dtype=float32)
        vals = _np.array(self.vals.data, copy=False).view(dtype=float32)
        locs = _np.array(self.locs, copy=False)
        threshhold = float(self.threshhold[0])
        count = _np.array(self.count, copy=False)
        inline(self.thecode, ['inarr', 'vals', 'locs', 'threshhold', 'count'],
               extra_compile_args=['-march=native -fprefetch-loop-arrays -funroll-loops -O3 -w'] + omp_flags,
               support_code = new_support,
               auto_downcast = 1,
               libraries=omp_libs)

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

_libthresh = libutils.get_ctypes_library('thresh', [])
if _libthresh is None:
    HAVE_LIBTHRESH = False
else:
    HAVE_LIBTHRESH = True

class CT_CountProblem(BaseThreshProblem):
    def __init__(self, size):
        if not HAVE_LIBTHRESH:
            raise RuntimeError("Cannot find libthresh.so")
        if size != 1048576:
            raise RuntimeError("This is a hacky code :) It only accepts size 1048576")
        super(CT_CountProblem, self).__init__(size=size)
        self._efunc = _libthresh.count_thresh
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_ulong, ctypes.c_float,
                                ctypes.c_ulong, ctypes.c_ulong]
        
        self.tsqr = _np.array([self.threshhold[0]*self.threshhold[0]],dtype=float32)
        self.start = _np.array([0], dtype=_np.uint32)
        self.chunkno = _np.array([0], dtype=_np.uint32)
        self.end = _np.array([131072], dtype=_np.uint32)
        # Following hack is to ensure alignment
        self.count_float = zeros(8, dtype=float32)
        self.countarr = _np.array(self.count_float.data, copy=False).view(dtype = _np.uint32)
        self.cntptr = self.countarr.ctypes.data
        self.inptr = self.input.ptr

    def execute(self):
        self._efunc(self.inptr, self.cntptr, self.chunkno[0], self.tsqr[0],
                    self.start[0], self.end[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


class CT_SumProblem(BaseThreshProblem):
    def __init__(self, size):
        if not HAVE_LIBTHRESH:
            raise RuntimeError("Cannot find libthresh.so")
        if size != 1048576:
            raise RuntimeError("This is a hacky code :) It only accepts size 1048576")
        super(CT_SumProblem, self).__init__(size=size)
        self._efunc = _libthresh.excl_prefix_sum
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        
        # Following hack is to ensure alignment
        self.count_float = zeros(8, dtype=float32)
        self.countarr = _np.array(self.count_float.data, copy=False).view(dtype = _np.uint32)
        self.cntptr = self.countarr.ctypes.data
        self.cntlen = _np.array([8], dtype = _np.uint32)

    def execute(self):
        self._efunc(self.cntptr, self.cntlen[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

###

class CT_CopyProblem(BaseThreshProblem):
    def __init__(self, size):
        if not HAVE_LIBTHRESH:
            raise RuntimeError("Cannot find libthresh.so")
        if size != 1048576:
            raise RuntimeError("This is a hacky code :) It only accepts size 1048576")
        super(CT_CopyProblem, self).__init__(size=size)
        self._efunc = _libthresh.copy_above
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_ulong, ctypes.c_float,
                                ctypes.c_ulong, ctypes.c_ulong]
        self.tsqr = _np.array([self.threshhold[0]*self.threshhold[0]],dtype=float32)
        self.start = _np.array([0], dtype=_np.uint32)
        self.chunkno = _np.array([0], dtype=_np.uint32)
        self.end = _np.array([131072], dtype=_np.uint32)
        self.inptr = self.input.ptr
        self.vptr = self.vals.ptr
        self.lptr = self.locs.ctypes.data        
        # Following hack is to ensure alignment
        self.count_float = zeros(8, dtype=float32)
        self.countarr = _np.array(self.count_float.data, copy=False).view(dtype = _np.uint32)
        self.cntptr = self.countarr.ctypes.data
        self.cntlen = _np.array([8], dtype = _np.uint32)
        self._cfunc = _libthresh.count_thresh
        self._cfunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_ulong, ctypes.c_float,
                                ctypes.c_ulong, ctypes.c_ulong]
        self._cfunc(self.inptr, self.cntptr, self.chunkno[0], self.tsqr[0],
                    self.start[0], self.end[0])
        self._sfunc = _libthresh.excl_prefix_sum
        self._sfunc.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._sfunc(self.cntptr, self.cntlen[0])


    def execute(self):
        self._efunc(self.inptr, self.vptr, self.lptr, self.cntptr,
                    self.chunkno[0], self.tsqr[0], self.start[0], self.end[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

##

CHUNKSIZE = 131072

class CT_BatchProblem(BaseThreshProblem):
    def __init__(self, size):
        if not HAVE_LIBTHRESH:
            raise RuntimeError("Cannot find libthresh.so")
        if size != 1048576:
            raise RuntimeError("This is a hacky code :) It only accepts size 1048576")
        super(CT_BatchProblem, self).__init__(size=size)

        self._efunc = _libthresh.thresh_many
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_float]
        self.inptr = self.input.ptr
        self.valptrs = _np.zeros(8, dtype = _np.uint64)
        self.locptrs = _np.zeros(8, dtype = _np.uint64)
        self.cntptrs = _np.zeros(8, dtype = _np.uint64)
        self.vallist = []
        self.loclist = []
        for i in range(8):
            newval = zeros(CHUNKSIZE, dtype=complex64)
            newloc = _np.zeros(CHUNKSIZE, dtype = _np.uint32)
            self.vallist.append(newval)
            self.loclist.append(newloc)
            self.valptrs[i] = newval.ptr
            self.locptrs[i] = newloc.ctypes.data
        self.vptr = self.valptrs.ctypes.data
        self.lptr = self.locptrs.ctypes.data
        self.cptr = self.cntptrs.ctypes.data

    def execute(self):
        self._efunc(self.inptr, self.vptr, self.lptr, self.cptr, self.threshhold[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()

class CT_Thresh2Problem(BaseThreshProblem):
    def __init__(self, size):
        if not HAVE_LIBTHRESH:
            raise RuntimeError("Cannot find libthresh.so")
        if size != 1048576:
            raise RuntimeError("This is a hacky code :) It only accepts size 1048576")
        super(CT_Thresh2Problem, self).__init__(size=size)

        self._efunc = _libthresh.thresh2
        self._efunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_float]
        self.inptr = self.input.ptr
        self.vptr = self.vals.ptr
        self.lptr = self.locs.ctypes.data

    def execute(self):
        self._efunc(self.inptr, self.vptr, self.lptr, self.threshhold[0])

    def _setup(self):
        # To setup, we just run once, which compiles and caches the code
        self.execute()


_class_dict = { 'numpy' : NumpyThreshProblem,
                'weave' : WeaveThreshProblem,
                'static' : StaticThreshProblem,
                'ct_count' : CT_CountProblem,
                'ct_sum' : CT_SumProblem,
                'ct_copy' : CT_CopyProblem,
                'ct_batch' : CT_BatchProblem,
                'ct_thresh2' : CT_Thresh2Problem,
                'new_pycbc' : NewPyCBCThreshProblem
               }

thresh_valid_methods = _class_dict.keys()

def parse_thresh_problem(probstring, method='numpy'):
    """
    This function takes a string of the form <number>

    It also takes another argument, a string indicating which class
    type to return. 

    It returns the class and size, so that the call:
        MyClass, n = parse_thresh_problem(probstring, method)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
