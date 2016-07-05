#ifndef _Vector_hpp_
#define _Vector_hpp_

//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#include <vector>

#include <qthread/qloop.h>
#include "qthreads_loop_type.h"
#include <MemInitOp.hpp>

namespace miniFE {

typedef struct {
	MINIFE_SCALAR* x;
} zerovect_thread;

void zerovect_thread_func(size_t start, size_t stop, void* thr_args) {
	zerovect_thread* thread_args = (zerovect_thread*) thr_args;
	MINIFE_SCALAR* x = thread_args->x;

	for(size_t i = start; i < stop; i++) {
		x[i] = 0;
	}
}

template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal>
struct Vector {
  typedef Scalar ScalarType;
  typedef LocalOrdinal LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;

  Vector(GlobalOrdinal startIdx, LocalOrdinal local_sz)
   : startIndex(startIdx),
     local_size(local_sz),
     coefs(local_size)
  {
    zerovect_thread thread_args = { &coefs[0] };
    QLOOP(0, local_size, zerovect_thread_func, &thread_args);
  }

  ~Vector()
  {
  }

  GlobalOrdinal startIndex;
  LocalOrdinal local_size;
  std::vector<Scalar> coefs;
};


}//namespace miniFE

#endif

