#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
#MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -v -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
	--cuda-path=$(CUDA_ROOT) -ffp-contract=fast -fopenmp-nonaliased-maps
CXXFLAGS = $(CFLAGS)

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) \
	$(MINIFE_MATRIX_TYPE) #\
#	-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK \
#	-DMINIFE_REPORT_RUSAGE

LDFLAGS=$(CFLAGS)
LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

#CXX=mpiicpc
#CC=mpiicc

#CXX=g++
#CC=g++

#CXX=icpc
#CC=icc

CXX=clang++
CC=clang

include make_targets
