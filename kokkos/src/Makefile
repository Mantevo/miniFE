#-----------------------------------------------------------------------
SHELL = /bin/sh

MPIPATH = /opt/mpi


CXX = mpicxx 
CC = mpicc
LINK = mpicxx 

#Note: when using absolute paths KOKKOSPATH and KOKKOSPATH_INC should be the same
#KOKKOSPATH is using for copying source files into the Object directory
#KOKKOSPATH_INC is used during the actual compilation
KOKKOSPATH = ../kokkos
KOKKOSPATH_INC = ../../kokkos

PWD = `pwd`

MINIFE_TYPES =  \
	-DMINIFE_SCALAR=double   \
	-DMINIFE_LOCAL_ORDINAL=int      \
	-DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
# MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX
 
CPPFLAGS = -O3 -I. -mavx -I../ -I../../utils -I../../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DMPICH_IGNORE_CXX_SEEK -I$(KOKKOSPATH_INC)/core/src -I$(KOKKOSPATH_INC)/containers/src -I$(KOKKOSPATH_INC)/linalg/src -fPIC 
LINKFLAGS = -O3 -mavx   

#Use MPI
CPPFLAGS += -DHAVE_MPI -I$(MPIPATH)/include

##Enable DEBUG
#CPPFLAGS += -g -G -DKOKKOSARRAY_EXPRESSION_CHECK -DENABLE_TRACEBACK
#LINKFLAGS += -g

#Enable Single Precision
#CPPFLAGS += -DPRECISION=1

SRC = $(shell ls *.cpp;)
CPY = $(PWD)/*.cpp

KOKKOS_SRC = $(shell cd $(KOKKOSPATH)/core/src/impl; ls *.cpp;)
KOKKOS_CPY = $(KOKKOSPATH)/core/src/impl/*.cpp

#Use OpenMP backend
KOKKOS_SRC += $(shell cd $(KOKKOSPATH)/core/src/OpenMP; ls *.cpp;)
KOKKOS_CPY += $(KOKKOSPATH)/core/src/OpenMP/*.cpp
CPPFLAGS += -DKOKKOS_HAVE_OPENMP -fopenmp
LINKFLAGS += -fopenmp 

##Use PThreads Backend
#KOKKOS_SRC += $(shell cd $(KOKKOSPATH)/core/src/Threads; ls *.cpp;)
#KOKKOS_CPY += $(KOKKOSPATH)/core/src/Threads/*.cpp
#CPPFLAGS += -DKOKKOS_HAVE_PTHREAD 
#USRLIB += -lpthread

##Use HWLOC
#HWLOCPATH = ./
#CPPFLAGS += -DKOKKOS_HAVE_HWLOC -I$(HWLOCPATH)/include
#USRLIB += -L$(HWLOCPATH)/lib -lhwloc 

#-----------------------------------------------------------------------
OBJ = $(KOKKOS_SRC:.cpp=.o)

SYSLIB = $(LIBMPI) $(INTELLIB) $(LIBIB)


# For debugging, the macro MINIFE_DEBUG will cause miniFE to dump a log file
# from each proc containing various information.
# This macro will also enable a somewhat expensive range-check on indices in
# the exchange_externals function.

LDFLAGS = $(LINKFLAGS)
LIBS= $(USRLIB) $(SYSLIB)

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.


copy: 
	@if [ ! -d Obj ]; then mkdir Obj; fi
	@echo '' > Obj/KokkosCore_config.h
	@cp -p $(SRC) Obj
	@cp -p $(KOKKOS_CPY) Obj
	@cp Makefile Obj/Makefile
	@cd Obj; ../get_common_files
	@cd Obj; $(MAKE) all "OBJ = $(OBJ)"


OBJ += BoxPartition.o YAML_Doc.o YAML_Element.o
OBJ += param_utils.o utils.o  mytimer.o
OBJ += main.o


MINIFE_INFO = 1
MINIFE_KERNELS = 0

vpath %.cpp ../../utils

all:generate_info miniFE.x

miniFE.x:$(OBJ) $(MAIN_OBJ) $(OBJS) $(UTIL_OBJS) $(OPTIONAL_OBJS) ../*.hpp generate_info
	$(INSTRUMENT) $(LINK) $(CXXFLAGS) $(OBJ) $(MAIN_OBJ) $(OBJS) $(UTIL_OBJS) $(OPTIONAL_OBJS) -o ../miniFE.x $(LDFLAGS) $(OPTIONAL_LIBS) $(LIBS)

generate_info:
	./generate_info_header "$(CXX)" "$(CXXFLAGS)" "miniFE" "MINIFE"

test:
	./run_test x

%.o:%.cpp *.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DMINIFE_INFO=$(MINIFE_INFO) -DMINIFE_KERNELS=$(MINIFE_KERNELS) -c $<

%.o:%.c *.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

clean:
	rm -rf *.o *.a miniFE.x *.linkinfo miniFE_info.hpp Obj

realclean: clean
	rm -f gmon.out gprof.* *~ *.yaml *.TVD.* *.mtx* *.vec* minife_debug*
  
