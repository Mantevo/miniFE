#@HEADER
# ************************************************************************
# 
#               Mantevo: A collection of mini-applications for HPC
#                 Copyright (2008) Sandia Corporation
# 
# Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
# license for use of this work by or on behalf of the U.S. Government.
# 
# This library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version.
#  
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#  
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
# USA
# Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
# 
# ************************************************************************
#@HEADER


# Simple hand-tuned makefile.  Modify as necessary for your environment.
# Questions? Contact Mike Heroux (maherou@sandia.gov).
#

#
# 0) Specify compiler and linker:

#CXX=g++
#LINKER=g++
CXX=mpic++
LINKER=mpic++


# 1) Build with MPI or not?
#    If you want to run the program with MPI, make sure USE_MPI is set 
#    to -DHAVE_MPI

#USE_MPI =
USE_MPI = -DHAVE_MPI


# 2) MPI headers:  
#    If you:
#    - Are building MPI mode (-DUSING_MPI is set above).
#    - Do not have the MPI headers installed a default search directory and
#    - Are not using MPI compiler wrappers
#    Then specify the path to your MPI header file (include a -I)

#MPI_INC = -I/usr/MPICH/SDK.gcc/include


# 3) Specify C++ compiler optimization flags (if any)
#    Typically some reasonably high level of optimization should be used to 
#    enhance performance.

#IA32 with GCC: 
#CPP_OPT_FLAGS = -O3 -funroll-all-loops -malign-double
#IA64 with GCC:
CPP_OPT_FLAGS = -O3 -funroll-all-loops
#DEBUG
#CPP_OPT_FLAGS = -O0 -g

#
# 4) MPI library:
#    If you:
#    - Are building MPI mode (-DUSING_MPI is set above).
#    - Do not have the MPI library installed a default search directory and
#    - Are not using MPI compiler wrappers for linking
#    Then specify the path to your MPI library (include -L and -l directives)

#MPI_LIB = -L/usr/MPICH/SDK.gcc/lib -lmpich

#
# 5) System libraries: (May need to add -lg2c before -lm)

SYS_LIB =-lm

#
# 6) Specify name if executable (optional):

TARGET = NewMiniApp.exe

################### Derived Quantities (no modification required) ##############

CXXFLAGS= $(CPP_OPT_FLAGS) $(USE_MPI) $(MPI_INC) 

LIB_PATHS= $(MPI_LIB) $(SYS_LIB)

TEST_CPP = NewMiniApp.cpp mytimer.cpp YAML_Doc.cpp YAML_Element.cpp

TEST_OBJ          = $(TEST_CPP:.cpp=.o)

$(TARGET): $(TEST_OBJ)
	$(LINKER) $(CFLAGS) $(TEST_OBJ) $(LIB_PATHS) -o $(TARGET)

clean:
	@rm -f *.o  *~ $(TARGET) $(TARGET).exe
