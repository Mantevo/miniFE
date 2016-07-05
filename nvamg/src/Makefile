#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
#NVAMG_PATH = /home/crtrott/Software/nvamg/2013-09-13/
NVAMG_PATH = /home/projects/nvidia/nvamg/2013-09-13/
#MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3 
CXXFLAGS = $(CFLAGS)

CPPFLAGS = -g -I. -I../utils -I../fem $(MINIFE_TYPES) -I$(NVAMG_PATH)/examples\
	$(MINIFE_MATRIX_TYPE) \
	-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK \
	-DMINIFE_REPORT_RUSAGE

LDFLAGS=$(CFLAGS) -L$(NVAMG_PATH)/lib -lnvamgsh
LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

#CXX=mpiicpc
#CC=mpiicc

#CXX=g++
#CC=g++

#CXX=icpc
#CC=icc

CXX=mpicxx
CC=mpicc

include make_targets
