#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <x86intrin.h>

void count_thresh(const float* __restrict__ input, 
                  unsigned int* __restrict__ count_arr, 
                  const unsigned int chunkno, const float tsqr, 
                  const unsigned int start, const unsigned int stop){

  // This function passes through the data first, counting the number
  // of points in each parallel segment that are above threshhold.
  //
  // Note that we pass in the squared value of the threshhold.

  unsigned int i, cnt, sc, ec;
  float re, im;

  // We make copies of 'start' and 'end' since we want them private
  // to the thread:

  sc = start;
  ec = stop;

  cnt = 0;
  asm("#Start count loop");
  for (i = sc; i < ec; i++){
    re = input[2*i];
    im = input[2*i+1];
    if ( (re*re + im*im) > tsqr){
      cnt++;
    }
  }
  asm("#End count loop");
  count_arr[chunkno] = cnt;
  return;
}

unsigned int excl_prefix_sum(unsigned int* __restrict__ count_arr, const unsigned int len){

  unsigned int i, inim1, tmp;

  // This function does a (serial) exclusive prefix sum on the array count_arr, in-place.
  // It returns the sum of *all* elements in the array (which would conceptually be the
  // value that would be just past the end of the exclusive sum array). 

  inim1 = count_arr[0];
  count_arr[0] = 0;
  asm("#Start prefix loop");  
  for (i = 1; i < len; i++){
    tmp = count_arr[i];
    count_arr[i] = inim1 + count_arr[i-1];
    inim1 = tmp;
  }
  asm("#End prefix loop");

  return (count_arr[len-1] + inim1);

}

unsigned int rw_excl_prefix_sum(const unsigned int* __restrict__ input,
				unsigned int* __restrict__ output,
				const unsigned int len){

  unsigned int i,  tmp;

  // This function does a (serial) exclusive prefix sum on the array count_arr, in-place.
  // It returns the sum of *all* elements in the array (which would conceptually be the
  // value that would be just past the end of the exclusive sum array). 

  output[0] = 0;
  for (i = 1; i < len; i++){
    output[i] = input[i-1] + output[i-1];
  }

  return (input[len-1] + output[len-1]);

}

void copy_above(const float* __restrict__ input, 
                float* __restrict__ vals,
                unsigned int * __restrict__ locs, 
                const unsigned int * __restrict__ count_arr,
                const unsigned int chunkno, const float tsqr, 
                const unsigned int start, const unsigned int stop) {

 // This function passes through the data once more, using the number of points above threshhold
 // in this segment to know (deterministically) where to write the values and locations of those
 // points that are above threshhold.  It is *critical* to the correct behavior of the overall
 // algorithm that the segmentation and also the matching between a segment and threadno be the same
 // for this function as for the preceding two.

  unsigned int offset, i, c, sc, ec;
  float re, im;

  // We make copies of 'start' and 'end' since we want them private
  // to the thread:

  sc = start;
  ec = stop;

  offset = count_arr[chunkno];

  c = 0;
  asm("#Start copy loop");
  for (i = start; i < stop; i++){
    re = input[2*i];
    im = input[2*i+1];
    if ((re*re + im*im) > tsqr){
      vals[2*(offset + c)] = re;
      vals[2*(offset + c) + 1] = im;
      locs[offset + c] = i;
      c++;
    }
  }
  asm("#End copy loop");
}

// Hard-coded for testing

#define NCHUNKS 8
#define NLEN 1048576
#define CHUNKSIZE 131072

unsigned int thresh(const float* __restrict__ inarr, float* __restrict__  vals, 
		    unsigned int* __restrict__ locs, const float threshhold){
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
  unsigned int startpts[NCHUNKS], endpts[NCHUNKS], i, count;
  float tsqr;

  tsqr = threshhold * threshhold;

  for (i = 0; i < NCHUNKS-1; i++){
    startpts[i] = i * CHUNKSIZE;
    endpts[i] = (i+1) * CHUNKSIZE;
  }
  startpts[NCHUNKS-1] = (NCHUNKS-1)*CHUNKSIZE;
  endpts[NCHUNKS-1] = NLEN;

#pragma omp parallel for schedule(static, 1)
  for (i = 0; i < NCHUNKS; i++){
    count_thresh(inarr, chunk_counts, i, tsqr, startpts[i], endpts[i]);  
  }

  count = excl_prefix_sum(chunk_counts, NCHUNKS);

#pragma omp parallel for schedule(static, 1)
  for (i = 0; i < NCHUNKS; i++){
    copy_above(inarr, vals, locs, chunk_counts, i, tsqr, startpts[i], endpts[i]); 
  }

  return count;
}

void copy_batch(const float* __restrict__ inarr,
		float* __restrict__ vals,
		unsigned int* __restrict__ locs,
		unsigned int chunkno,
		unsigned int* __restrict__ count,
		unsigned int chunksize,
		float tsqr){

  unsigned int i, c, offset;
  float re, im, tval;

  // Copy to help OpenMP
  tval = tsqr;
  offset = chunkno*chunksize;

  c = 0;
  for (i = 0; i < chunksize; i++){
    re = inarr[2*i];
    im = inarr[2*i+1];
    if ((re*re + im*im) > tval){
      vals[2*c] = re;
      vals[2*c+1] = im;
      locs[c] = i+offset;
      c++;
    }
  }

  *count = c;
  return;
}

// What if we parallelize differently?  Create several output vectors,
// hand those off, and then we don't need but one pass over the data.
// But that depends on what the cost is to put things back together
// in Python

unsigned int thresh_many(const float* __restrict__ inarr, float* __restrict__  vals[NCHUNKS], 
			 unsigned int* __restrict__ locs[NCHUNKS], unsigned int counts[NCHUNKS],
			 const float threshhold){

  unsigned int i;
  float tsqr;

  tsqr = threshhold * threshhold;

#pragma omp parallel for schedule(static, 1)
  for (i = 0; i < NCHUNKS; i++){
    copy_batch(&inarr[i*CHUNKSIZE], vals[i], locs[i], i, &counts[i], CHUNKSIZE, tsqr); 
  }

  return;
}

unsigned int thresh2(const float* __restrict__ inarr, float* __restrict__  vals, 
		     unsigned int* __restrict__ locs, const float threshhold){

  unsigned int i, counts[NCHUNKS], psumcnts[NCHUNKS], *tmplocs[NCHUNKS], cnt;
  float *tmpvals[NCHUNKS];

  for (i = 0; i < NCHUNKS; i++){
    tmplocs[i] = malloc(CHUNKSIZE*sizeof(unsigned int));
    tmpvals[i] = malloc(CHUNKSIZE*sizeof(float));
  }

  thresh_many(inarr, tmpvals, tmplocs, counts, threshhold);
  cnt = rw_excl_prefix_sum(counts, psumcnts, NCHUNKS);
#pragma omp parallel for schedule(static, 1)
  for (i = 0; i < NCHUNKS; i++){
    memcpy(&vals[2*psumcnts[i]], tmpvals[i], 2*counts[i]*sizeof(float));
    memcpy(&locs[psumcnts[i]], tmplocs[i], counts[i]*sizeof(unsigned int));
    free(tmpvals[i]);
    free(tmplocs[i]);
  }

  return cnt;
}

// Now some attempts at AVX optimization

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif


void count_thresh_avx(const float* __restrict__ input, 
		      unsigned int* __restrict__ count_arr, 
		      const unsigned int chunkno, const float tsqr, 
		      const unsigned int start, const unsigned int stop){

#if _HAVE_AVX

  return;


#else
#error AVX unavailable
#endif



}
