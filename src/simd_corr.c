#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

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

void ccmul(float * __restrict a, float * __restrict b, float * __restrict c, int N){

  int i;

#if _HAVE_AVX

  /*
    __builtin_assume_aligned(a, 32);
    __builtin_assume_aligned(b, 32);
    __builtin_assume_aligned(c, 32);
  */

  float minus_ones_vec[8] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  __m256 minus_ones;

  minus_ones = _mm256_loadu_ps(minus_ones_vec);

  // Unroll three times.  Interleave slightly differently to avoid need for
  // multiple "minus_ones" registers.
  asm("#Start loop");
  for (i = 0; i < ROUND_DOWN(N, 24); i += 24){
    __m256 x1, y1, z1, areg1, breg1;
    __m256 x2, y2, z2, areg2, breg2;
    __m256 x3, y3, z3, areg3, breg3;

    // prefetch next loop's iterations in memory

    _mm_prefetch(a+i+24, 1);
    _mm_prefetch(b+i+24, 1);
    _mm_prefetch(c+i+24, 1);

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
  asm("#End loop");

#elif _HAVE_SSE3

  // We load and store in reverse order, to avoid
  // a multiplication by -1 implied by the conjugation
  // of array 'a'.

  /*
    __builtin_assume_aligned(a, 16);
    __builtin_assume_aligned(b, 16);
    __builtin_assume_aligned(c, 16);
  */

  // Unroll three times

  for (i = 0; i < ROUND_DOWN(N, 12); i += 12){
    __m128 arev1, brev1, x1, y1, z1;
    __m128 arev2, brev2, x2, y2, z2;
    __m128 arev3, brev3, x3, y3, z3;

    // prefetch next loop's iterations in memory

    _mm_prefetch(a+i+12, 1);
    _mm_prefetch(b+i+12, 1);
    _mm_prefetch(c+i+12, 1);

    // Load everything into registers

    arev1 = _mm_loadr_ps(a+i);
    arev2 = _mm_loadr_ps(a+i+4);
    arev3 = _mm_loadr_ps(a+i+8);

    brev1 = _mm_loadr_ps(b+i);
    brev2 = _mm_loadr_ps(b+i+4);
    brev3 = _mm_loadr_ps(b+i+8);

    x1 = _mm_movehdup_ps(arev1);
    x2 = _mm_movehdup_ps(arev2);
    x3 = _mm_movehdup_ps(arev3);

    z1 = _mm_mul_ps(brev1, x1);
    z2 = _mm_mul_ps(brev2, x2);
    z3 = _mm_mul_ps(brev3, x3);

    x1 = _mm_moveldup_ps(arev1);
    x2 = _mm_moveldup_ps(arev2);
    x3 = _mm_moveldup_ps(arev3);

    y1 = _mm_shuffle_ps(brev1, brev1, 0xB1);
    y2 = _mm_shuffle_ps(brev2, brev2, 0xB1);
    y3 = _mm_shuffle_ps(brev3, brev3, 0xB1);

    y1 = _mm_mul_ps(x1, y1);
    y2 = _mm_mul_ps(x2, y2);
    y3 = _mm_mul_ps(x3, y3);

    x1 = _mm_addsub_ps(z1, y1);
    x2 = _mm_addsub_ps(z2, y2);
    x3 = _mm_addsub_ps(z3, y3);

    _mm_storer_ps(c+i, x1);
    _mm_storer_ps(c+i+4, x2);
    _mm_storer_ps(c+i+8, x3);

  }

#else
#error Neither AVX nor SSE3 available
#endif

}
