#-----------------------------------------------------------------------

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
# MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

QTHREADS_LOOP_TYPE = USING_QLOOP_QT_LOOP_BALANCE
#QTHREADS_LOOP_TYPE = USING_QLOOP_QT_LOOP_QUEUE_CHUNK
#QTHREADS_LOOP_TYPE = USING_QLOOP_QT_LOOP_QUEUE_GUIDED
#QTHREADS_LOOP_TYPE = USING_QLOOP_QT_LOOP_QUEUE_FACTORED
#QTHREADS_LOOP_TYPE = USING_QLOOP_QT_LOOP_QUEUE_TIMED
#QTHREADS_LOOP_TYPE = USING_QLOOP_QT_LOOP

#-----------------------------------------------------------------------

CFLAGS = -O3 -I$(QTHREADS)/include -mavx -D$(QTHREADS_LOOP_TYPE)
CXXFLAGS = $(CFLAGS)

# For debugging, the macro MINIFE_DEBUG will cause miniFE to dump a log file
# from each proc containing various information.
# This macro will also enable a somewhat expensive range-check on indices in
# the exchange_externals function.

# CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) -DMINIFE_DEBUG -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK
CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) #-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK

LDFLAGS=-L$(QTHREADS)/lib
LIBS=-lqthread -pthread -lrt -lhwloc

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

CXX=icpc
CC=icc

include make_targets

