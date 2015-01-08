#include <stdlib.h>
#include <omp.h>
#include <complex.h>
#include <math.h>

int correlate_dynf(complex float *x, complex float *y, complex float *z, int N){
  int i;
  float xr, xi, yr, yi, zr, zi;

#pragma omp parallel for
  for (i = 0; i  < N; i++){
    xr = crealf(x[i]);
    xi = cimagf(x[i]);
    yr = crealf(y[i]);
    yi = cimagf(y[i]); 
    zr = xr*yr +xi*yi;
    zi = xr*yi - xi*yr;
    z[i] = zr + I*zi;
    //z[i] = conjf(x[i])*y[i];
  }
  return 0;

}

int correlate_dyn(complex double *x, complex double *y, complex double *z, int N){
  int i;

#pragma omp parallel for
  for (i = 0; i  < N; i++){
    z[i] = conj(x[i])*y[i];
  }
  return 0;

}

#define CHUNKSIZE 131072

int correlate_staticf(complex float *x, complex float *y, complex float *z, int N){
  int i;
  float xr, xi, yr, yi, zr, zi;
  
#pragma omp parallel for schedule(static, CHUNKSIZE)
  for (i = 0; i  < N; i++){
    xr = crealf(x[i]);
    xi = cimagf(x[i]);
    yr = crealf(y[i]);
    yi = cimagf(y[i]); 
    zr = xr*yr +xi*yi;
    zi = xr*yi - xi*yr;
    z[i] = zr + I*zi;
    //z[i] = conjf(x[i])*y[i];
  }
  return 0;

}

int correlate_static(complex double *x, complex double *y, complex double *z, int N){
  int i;

#pragma omp parallel for schedule(static, CHUNKSIZE)
  for (i = 0; i  < N; i++){
    z[i] = conj(x[i])*y[i];
  }
  return 0;

}

int correlate_stackedf(complex float *x, int N){
  int i;
  float xr, xi, yr, yi, zr, zi;
  
#pragma omp parallel for schedule(static, CHUNKSIZE)
  for (i = 0; i  < N; i++){
    xr = crealf(x[i]);
    xi = cimagf(x[i]);
    yr = crealf(x[i+CHUNKSIZE]);
    yi = cimagf(x[i+CHUNKSIZE]); 
    zr = xr*yr +xi*yi;
    zi = xr*yi - xi*yr;
    x[i+2*CHUNKSIZE] = zr + I*zi;
    //z[i] = conjf(x[i])*y[i];
  }
  return 0;

}
