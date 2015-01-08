#include <stdlib.h>
#include <complex.h>
#include <math.h>

int ccorr(complex float *x, complex float *y, complex float *z, int N){
  int i;
  complex float *xx, *yy, *zz;

  xx = __builtin_assume_aligned(x, 32);
  yy = __builtin_assume_aligned(y, 32);
  zz = __builtin_assume_aligned(z, 32);
  float xr, xi, yr, yi, zr, zi;
  asm("#Start loop");
  for (i = 0; i  < N; i++){
    xr = crealf(xx[i]);
    xi = cimagf(xx[i]);
    yr = crealf(yy[i]);
    yi = cimagf(yy[i]); 
    zr = xr*yr +xi*yi;
    zi = xr*yi - xi*yr;
    zz[i] = zr + I*zi;
  }
  asm("#End loop");
  return 0;

}
