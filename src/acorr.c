#include <stdlib.h>
#include <complex.h>
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

void scadd(float *a, float *b, float *c, int N){

  int i;

#if _HAVE_AVX

  __m256 a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4;

  // Unroll four times.
  asm("#Start loop");
  for (i = 0; i < ROUND_DOWN(N, 32); i += 32){

    // prefetch next loop's iterations in memory

    _mm_prefetch(a+i+32, 1);
    _mm_prefetch(b+i+32, 1);
    _mm_prefetch(c+i+32, 1);

    // Load everything into registers

    a1 = _mm256_load_ps(a+i);
    a2 = _mm256_load_ps(a+i+8);
    a3 = _mm256_load_ps(a+i+16);
    a3 = _mm256_load_ps(a+i+24);
    b1 = _mm256_load_ps(b+i);
    b2 = _mm256_load_ps(b+i+8);
    b3 = _mm256_load_ps(b+i+16);
    b3 = _mm256_load_ps(b+i+24);

    c1 = _mm256_add_ps(a1, b1);
    c2 = _mm256_add_ps(a2, b2);
    c3 = _mm256_add_ps(a3, b3);
    c4 = _mm256_add_ps(a4, b4);

    _mm256_store_ps(c+i, c1);
    _mm256_store_ps(c+i+8, c2);
    _mm256_store_ps(c+i+16, c3);
    _mm256_store_ps(c+i+24, c4);
  }
  asm("#End loop");

  return;
#else
#error AVX not available
#endif

}

void cadd(complex float *a, complex float *b, complex float *c, int N){

  int i;

  asm("#Start 2nd loop");
  for (i = 0; i < N; i++){
    c[i] = a[i] + b[i];
  }
  asm("#End 2nd loop");

  return;
}
