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
import pycbc.fft
from pycbc import scheme as _scheme
import pycbc.fft.mkl as _mkl
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils
from math import sqrt
import sys

# Several of the OpenMP based approaches use this
max_chunk = 8192

_mkl_lib = _mkl.lib
_create_descr = _mkl_lib.DftiCreateDescriptor
_create_descr.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]


def plan_batched(nsize, nbatch=1, padding=0, inplace=False, nthreads=1):
    """
    A function to create an FFTW plan for a batched, strided,
    possibly multi-threaded transform.
    """

    desc = ctypes.c_void_p(1)
    prec = _mkl.DFTI_SINGLE
    domain = _mkl.DFTI_COMPLEX
    status = _create_descr(ctypes.byref(desc), prec, domain, 1, nsize)
    _mkl.check_status(status)
    # Now we set various things depending on exactly what kind of transform we're
    # performing.
    # In-place or out-of-place:
    if inplace:
        status = _mkl_lib.DftiSetValue(desc, _mkl.DFTI_PLACEMENT, _mkl.DFTI_INPLACE)
    else:
        status = _mkl_lib.DftiSetValue(desc, _mkl.DFTI_PLACEMENT, _mkl.DFTI_NOT_INPLACE)
    _mkl.check_status(status)

    # If we are performing a batched transform:
    if nbatch > 1:
        status = _mkl_lib.DftiSetValue(desc, _mkl.DFTI_NUMBER_OF_TRANSFORMS, nbatch)
        _mkl.check_status(status)
        status = _mkl_lib.DftiSetValue(desc, _mkl.DFTI_INPUT_DISTANCE, nsize+padding)
        _mkl.check_status(status)
        status = _mkl_lib.DftiSetValue(desc, _mkl.DFTI_OUTPUT_DISTANCE, nsize+padding)
        _mkl.check_status(status)

    # Knowing how many threads will be allowed may help select a better transform
    nthreads = _scheme.mgr.state.num_threads
    status = _mkl_lib.DftiSetValue(desc, _mkl.DFTI_THREAD_LIMIT, nthreads)
    _mkl.check_status(status)


    # Now everything's ready, so commit
    status = _mkl_lib.DftiCommitDescriptor(desc)
    _mkl.check_status(status)

    return desc

hand_fft_support = """
#include <complex>
#include <mkl.h>
#include <omp.h>

"""

hand_fft_libs = ['mkl']
# The following could return system libraries, but oh well,
# shouldn't hurt anything
hand_fft_link_args = ['-Wl,-rpath,/opt/intel/advisor_xe/lib64',
                      '-Wl,-rpath,/opt/intel/composerxe/ipp/lib/intel64',
                      '-Wl,-rpath,/opt/intel/composerxe/mkl/lib/intel64']

phase1_code = """
int j;

// First, execute half the FFTs for first phase
#pragma omp parallel for schedule(guided, 1)
for (j = 0; j < NJOBS1; j++){
   int tid = omp_get_thread_num();
   DftiComputeBackward((DFTI_DESCRIPTOR_HANDLE) plan1[tid], (complex *) &vin[j*NCHUNK1],
                      (complex *) &vout[j*NCHUNK1]);
}

"""

phase2_code = """
int j;

#pragma omp parallel for schedule(static, 1)
for (j = 0; j < NJOBS2; j++){
   int tid = omp_get_thread_num();
   DftiComputeBackward((DFTI_DESCRIPTOR_HANDLE) plan2[tid], (complex *) &vin[j*NCHUNK2],
                      (complex *) &vout[j*NCHUNK2]);
}

"""


def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

PAD_LEN = 8

class BaseHandFFTProblem(_mb.MultiBenchProblem):
    def __init__(self, size, inplace = False, padding = True):
        # We'll do some arithmetic with these, so sanity check first:
        if not check_pow_two(size):
            raise ValueError("Only power-of-two sizes supported")

        self.ncpus = _scheme.mgr.state.num_threads
        self.size = size
        self.nfirst = 2 ** int(_np.log2( size ) / 2)
        self.nsecond = size/self.nfirst

        if padding:
            self.padding = PAD_LEN
        else:
            self.padding = 0

        self.vsize = (self.nfirst + self.padding) * (self.nsecond + self.padding)
        self.invec = zeros(self.vsize, dtype = complex64)
        self.inplace = inplace
        if inplace:
            self.outvec = self.invec
        else:
            self.outvec = zeros(self.vsize, dtype = complex64)
        # Pointers are probably 64 bits; leave enough space just in case
        self.firstplans = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.secondplans = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.nbatch1 = max_chunk/self.nfirst
        self.nbatch2 = max_chunk/self.nsecond

    def _setup(self):
        # Our batched FFTs are executed using a single thread, since they will be called
        # from inside an OpenMP parallel region. We make a plan for each cpu, so that
        # there is not contention to read the plans (and they can stay in cache).
        for i in range(0, self.ncpus):
            self.firstplans[i] = plan_batched(self.nfirst, self.nbatch1, self.padding,
                                              inplace = self.inplace, nthreads = 1).value
            self.secondplans[i] = plan_batched(self.nsecond, self.nbatch2, self.padding,
                                               inplace = self.inplace, nthreads = 1).value
        self.execute()

# Now our derived classes

class PhaseOneProblem(BaseHandFFTProblem):
    def __init__(self, size, inplace = False, padding = True):
        super(PhaseOneProblem, self).__init__(size, inplace = inplace, padding = padding)
        tmpcode = phase1_code.replace('NJOBS1', str( self.size/(2*max_chunk) ) )
        self.code = tmpcode.replace('NCHUNK1', str( (self.nfirst + self.padding) * self.nbatch1))
        del tmpcode

    def execute(self):
        vin = _np.array(self.invec.data, copy = False)
        vout = _np.array(self.outvec.data, copy = False)
        plan1 = _np.array(self.firstplans, copy = False)
        inline(self.code, ['vin', 'vout', 'plan1'],
               extra_compile_args=['-openmp -xHOST -O3'],
               libraries = hand_fft_libs, compiler = 'icc',
               support_code = hand_fft_support, extra_link_args = hand_fft_link_args,
               verbose = 2, auto_downcast = 1)

class PhaseTwoProblem(BaseHandFFTProblem):
    def __init__(self, size, inplace = False, padding = True):
        super(PhaseTwoProblem, self).__init__(size, inplace = inplace, padding = padding)
        tmpcode = phase2_code.replace('NJOBS2', str( self.size/max_chunk ) )
        self.code = tmpcode.replace('NCHUNK2', str( (self.nsecond + self.padding) * self.nbatch2))
        del tmpcode

    def execute(self):
        vin = _np.array(self.invec.data, copy = False)
        vout = _np.array(self.outvec.data, copy = False)
        plan2 = _np.array(self.secondplans, copy = False)
        inline(self.code, ['vin', 'vout', 'plan2'],
               extra_compile_args=['-openmp -xHOST -O3'],
               libraries = hand_fft_libs, compiler = 'icc',
               support_code = hand_fft_support, extra_link_args = hand_fft_link_args,
               verbose = 2, auto_downcast = 1)

class PhaseOneInplaceProblem(BaseHandFFTProblem):
    def __init__(self, size, padding = True):
        super(PhaseOneInplaceProblem, self).__init__(size, inplace = True, padding = padding)

class PhaseTwoInplaceProblem(BaseHandFFTProblem):
    def __init__(self, size, padding = True):
        super(PhaseTwoInplaceProblem, self).__init__(size, inplace = True, padding = padding)

_class_dict = { 'phase_one' : PhaseOneProblem,
                'phase_two' : PhaseTwoProblem,
                'phase_one_inplace' : PhaseOneInplaceProblem,
                'phase_two_inplace' : PhaseTwoInplaceProblem
               }

mkl_hand_valid_methods = _class_dict.keys()

def parse_mkl_hand_problem(probstring, method):
    """
    This function takes a string of the form <number>


    It returns the class and size, so that the call:
        MyClass, n = parse_mkl_hand_problem(probstring)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
