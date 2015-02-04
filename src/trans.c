#include <stdlib.h>
#include <omp.h>

void Transpose(double* const A, const int n, const int* const plan ) {// double is float or double
  const int TILE = 32; // Tile size
  const int nEven = n - n%TILE; // nEven is a multiple of TILE
  const int wTiles = nEven / TILE; // Complete tiles in each dimens.
  const int nTilesParallel = wTiles*(wTiles - 1)/2; // # of complete tiles under the main diag.
#pragma omp parallel // Start of parallel region
  {
#pragma omp for schedule(guided)
    for (int k = 0; k < nTilesParallel; k++) { // Parallel loop over body tiles
      const int ii = plan[2*k + 0]; // Top column of the tile (planned)
      const int jj = plan[2*k + 1]; // Left row of the tile (planned)

      for (int j = jj; j < jj+TILE; j++) // Tile transposition microkernel:
#pragma simd // Ensure automatic vectorization
	for (int i = ii; i < ii+TILE; i++) {
	  const double c = A[i*n + j]; //
	  A[i*n + j] = A[j*n + i]; // Swap matrix elements
	  A[j*n + i] = c; //
	}
    } // End of main parallel for-loop

#pragma omp for schedule(static)
    for (int ii = 0; ii < nEven; ii += TILE) { // Transposing tiles on the main diagonal:
      const int ii = jj;
      for (int j = jj; j < jj+TILE; j++) // Diagonal tile transposition microkernel:
#pragma simd // Ensure automatic vectorization
	for (int i = ii; i < j; i++) { // Avoid duplicate swaps
	  const double c = A[i*n + j]; //
	  A[i*n + j] = A[j*n + i]; // Swap matrix elements
	  A[j*n + i] = c; //
	}
    }

#pragma omp for schedule(static)
    for (int j = 0; j < nEven; j++) // Transposing the "peel":
      for (int i = nEven; i < n; i++) {
	const double c = A[i*n + j]; //
	A[i*n + j] = A[j*n + i]; // Swap matrix elements
	A[j*n + i] = c; //
      }
  } // End of thread-parallel region

  for (int j = nEven; j < n; j++) // Transposing bottom-right cornr
    for (int i = nEven; i < j; i++) {
      const double c = A[i*n + j]; //
      A[i*n + j] = A[j*n + i]; // Swap matrix elements
      A[j*n + i] = c; //
    }
}
