#-----------------------------------------------------------------------
SHELL = /bin/sh

CXX = mpicxx 
CC = mpicc
LINK = mpicxx 

# Kokkos Settings:

KOKKOS_DEVICES = OpenMP
KOKKOS_ARCH = SNB

# MiniFE Settings:
MINIFE_TYPES =  \
	-DMINIFE_SCALAR=double   \
	-DMINIFE_LOCAL_ORDINAL=int      \
	-DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX
# MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

MINIFE_INFO = 1
MINIFE_KERNELS = 0


#PATHS

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

MINIFE_PATH = ${MAKEFILE_PATH}..

KOKKOS_PATH ?= ../kokkos
KOKKOS_PATH_INC ?= ${KOKKOS_PATH}

all:generate_info miniFE.x

CXXFLAGS = -O3 

LDFLAGS = -O3   


MINIFE_INCLUDES  = -I./ -I${MINIFE_PATH} -I${MINIFE_PATH}/src -I${MINIFE_PATH}/kokkos/linalg/src 
MINIFE_INCLUDES += -I${MINIFE_PATH}/fem -I${MINIFE_PATH}/utils -I${MINIFE_PATH}/common
override CXXFLAGS += -DMPICH_IGNORE_CXX_SEEK -fPIC $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) ${MINIFE_INCLUDES}  
override CXXFLAGS += -DMINIFE_INFO=$(MINIFE_INFO) -DMINIFE_KERNELS=$(MINIFE_KERNELS) -DUSE_MPI_WTIME

#Use MPI
override CXXFLAGS += -DHAVE_MPI

include $(KOKKOS_PATH)/Makefile.kokkos

#Enable Single Precision
#override CXXFLAGS += -DPRECISION=1

SRC = $(wildcard $(MINIFE_PATH)/src/*.cpp)
SRC += $(MINIFE_PATH)/common/YAML_Doc.cpp $(MINIFE_PATH)/common/YAML_Element.cpp
SRC += $(wildcard $(MINIFE_PATH)/utils/*.cpp)

HEADERS = $(wildcard $(MAKEFILE_PATH)/src/*.hpp)
HEADERS += $(MINIFE_PATH)/common/YAML_Doc.hpp $(MINIFE_PATH)/common/YAML_Element.hpp
HEADERS += $(wildcard $(MINIFE_PATH)/utils/*.hpp)

vpath %.cpp $(sort $(dir $(SRC)))

$(warning $(SRC))
OBJ = $(notdir $(SRC:.cpp=.o))
$(warning $(OBJ))
$(warning $(HEADERS))
$(warning $(KOKKOS_PATH))
$(warning $(KOKKOS_CPPFLAGS))

generate_info: 
	@${MINIFE_PATH}/common/generate_info_header "$(CXX)" "$(CXXFLAGS)" "miniFE" "MINIFE"
	

miniFE.x:$(OBJ) $(KOKKOS_LINK_DEPENDS) generate_info $(HEADERS)
	$(INSTRUMENT) $(LINK) $(LDFLAGS) $(KOKKOS_LDFLAGS) $(OBJ) $(KOKKOS_LIBS) -o miniFE.x

test:
	./run_test x

clean:
	rm -rf *.o *.a miniFE.x *.linkinfo miniFE_info.hpp Obj

realclean: clean
	rm -f gmon.out gprof.* *~ *.yaml *.TVD.* *.mtx* *.vec* minife_debug*
  
%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) -c $< -o $(notdir $@)
  