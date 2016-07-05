#ifndef _CSRMatrix_hpp_
#define _CSRMatrix_hpp_

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

#include <cstddef>
#include <vector>
#include <algorithm>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <Kokkos_Types.hpp>
#include <Kokkos_CrsMatrix.hpp>
#include <Kokkos_Vector.hpp>

namespace miniFE {

template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal, class Device = device_device_type>
struct
CSRMatrix {
  CSRMatrix()
   : has_local_indices(false),
     rows(), row_offsets(), row_offsets_external(),
     packed_cols(), packed_coefs(),
     num_cols(0)
#ifdef HAVE_MPI
     ,external_index(), external_local_index(), elements_to_send(),
      neighbors(), recv_length(), send_length(), send_buffer(), request()
#endif
  {
  }

  ~CSRMatrix()
  {}

  typedef Scalar        ScalarType;
  typedef LocalOrdinal  LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;
  typedef typename Kokkos::CrsMatrix<Scalar,int,Device,void,LocalOrdinal> Kokkos_Matrix;
  typedef CSRMatrix<Scalar,LocalOrdinal,GlobalOrdinal,host_device_type> HostMirror;
  typedef Device device_type;
  bool                       has_local_indices;
  Kokkos::vector<GlobalOrdinal,Device> rows;
  Kokkos::vector<LocalOrdinal,Device>  row_offsets;
  Kokkos::vector<LocalOrdinal,Device>  row_offsets_external;
  Kokkos::vector<GlobalOrdinal,Device> packed_cols;
  Kokkos::vector<Scalar,Device>        packed_coefs;
  LocalOrdinal               num_cols;

  Kokkos_Matrix mat;

  void init_kokkos_matrix() {
	  mat = Kokkos_Matrix("A",row_offsets.size()-1,num_cols,row_offsets(row_offsets.size()-1),packed_coefs.d_view,row_offsets.d_view,packed_cols.d_view);
  };

#ifdef HAVE_MPI
  Kokkos::vector<GlobalOrdinal,Device> external_index;
  Kokkos::vector<GlobalOrdinal,Device> external_local_index;
  Kokkos::vector<GlobalOrdinal,Device> elements_to_send;
  Kokkos::vector<int,Device>           neighbors;
  Kokkos::vector<LocalOrdinal,Device>  recv_length;
  Kokkos::vector<LocalOrdinal,Device>  send_length;
  Kokkos::vector<Scalar,Device>        send_buffer;
  std::vector<MPI_Request>   request;
#endif

  size_t num_nonzeros() const
  {
    return row_offsets[row_offsets.size()-1];
  }

  void reserve_space(unsigned nrows, unsigned ncols_per_row)
  {
    rows.resize(nrows);
    row_offsets.resize(nrows+1);
    packed_cols.reserve(nrows * ncols_per_row);
    packed_coefs.reserve(nrows * ncols_per_row);
  }

  void get_row_pointers(GlobalOrdinalType row, size_t& row_length,
                        GlobalOrdinalType*& cols,
                        ScalarType*& coefs) const
  {
    ptrdiff_t local_row = -1;
    //first see if we can get the local-row index using fast direct lookup:
    if (rows.size() >= 1) {
      ptrdiff_t idx = row - rows[0];
      if (idx < rows.size() && rows[idx] == row) {
        local_row = idx;
      }
    }
 
    //if we didn't get the local-row index using direct lookup, try a
    //more expensive binary-search:
    if (local_row == -1) {
      typename Kokkos::vector<GlobalOrdinal,Device>::iterator row_iter =
          std::lower_bound(rows.begin(), rows.end(), row);
  
      //if we still haven't found row, it's not local so jump out:
      if (row_iter == rows.end() || *row_iter != row) {
        row_length = 0;
        return;
      }
  
      local_row = row_iter - rows.begin();
    }

    LocalOrdinalType offset = row_offsets[local_row];
    row_length = row_offsets[local_row+1] - offset;
    cols = &packed_cols[offset];
    coefs = &packed_coefs[offset];
  }

  void host_to_device(){
	  rows.host_to_device();
	  row_offsets.host_to_device();
	  row_offsets_external.host_to_device();
	  packed_cols.host_to_device();
	  packed_coefs.host_to_device();

	#ifdef HAVE_MPI
	  external_index.host_to_device();
	  external_local_index.host_to_device();
	  elements_to_send.host_to_device();
	  neighbors.host_to_device();
	  recv_length.host_to_device();
	  send_length.host_to_device();
	  send_buffer.host_to_device();
	#endif
  }

  void on_device(){
	  rows.on_device();
	  row_offsets.on_device();
	  row_offsets_external.on_device();
	  packed_cols.on_device();
	  packed_coefs.on_device();

	#ifdef HAVE_MPI
	  external_index.on_device();
	  external_local_index.on_device();
	  elements_to_send.on_device();
	  neighbors.on_device();
	  recv_length.on_device();
	  send_length.on_device();
	  send_buffer.on_device();
	#endif
  }

};


}//namespace miniFE

#endif

