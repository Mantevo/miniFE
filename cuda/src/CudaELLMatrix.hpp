#ifndef _CUDAELLMatrix_hpp_
#define _CUDAELLMatrix_hpp_

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
#include <CudaUtils.h>
#include <thrust/device_vector.h>
#include <nvToolsExt.h>
namespace miniFE {

template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal>
struct
PODELLMatrix {
  typedef Scalar        ScalarType;
  typedef LocalOrdinal  LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;
 
  GlobalOrdinal               *rows;
  GlobalOrdinal               *cols;
  Scalar                      *coefs;
  int                         *external_map;
  GlobalOrdinal               num_rows;
  LocalOrdinal                num_cols_per_row;
  GlobalOrdinal               pitch;
};

template<typename Scalar,
         typename LocalOrdinal,
         typename GlobalOrdinal>
struct
CudaELLMatrix {
  CudaELLMatrix()
   : has_local_indices(false),
     rows(),
     cols(), coefs(),
     num_cols(0),
     num_cols_per_row(0)
#ifdef HAVE_MPI
     ,external_index(), external_local_index(), elements_to_send(),
      neighbors(), recv_length(), send_length(), request()
#ifndef GPUDIRECT 
     ,send_buffer()
#endif
#endif
  {
  }

  ~CudaELLMatrix()
  {
    cudaHostUnregister(&cols[0]);
    cudaCheckError();
  }

  void copyToDevice() const {
    int n=d_coefs.size();
    cudaMemcpy(const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[0])),const_cast<Scalar*>(&coefs[0]),sizeof(Scalar)*n,cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(const_cast<GlobalOrdinal*>(thrust::raw_pointer_cast(&d_cols[0])),const_cast<GlobalOrdinal*>(&cols[0]),sizeof(GlobalOrdinal)*n,cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(const_cast<GlobalOrdinal*>(thrust::raw_pointer_cast(&d_rows[0])),const_cast<GlobalOrdinal*>(&rows[0]),sizeof(GlobalOrdinal)*rows.size(),cudaMemcpyHostToDevice);
    cudaCheckError();
  }
  void copyToHost() const {
    int n=d_coefs.size();
    cudaMemcpy(const_cast<Scalar*>(&coefs[0]),const_cast<Scalar*>(thrust::raw_pointer_cast(&d_coefs[0])),sizeof(Scalar)*n,cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(const_cast<GlobalOrdinal*>(&cols[0]),const_cast<GlobalOrdinal*>(thrust::raw_pointer_cast(&d_cols[0])),sizeof(GlobalOrdinal)*n,cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(const_cast<GlobalOrdinal*>(&rows[0]),const_cast<GlobalOrdinal*>(thrust::raw_pointer_cast(&d_rows[0])),sizeof(GlobalOrdinal)*rows.size(),cudaMemcpyDeviceToHost);
    cudaCheckError();
  }

  PODELLMatrix<Scalar,LocalOrdinal,GlobalOrdinal> getPOD() {
    PODELLMatrix<Scalar,LocalOrdinal,GlobalOrdinal> ret;
    ret.rows=thrust::raw_pointer_cast(&d_rows[0]);
    ret.cols=thrust::raw_pointer_cast(&d_cols[0]);
    ret.coefs=thrust::raw_pointer_cast(&d_coefs[0]);
#ifdef HAVE_MPI
    ret.external_map=thrust::raw_pointer_cast(&d_external_map[0]);
#endif
    ret.num_cols_per_row=num_cols_per_row;
    ret.num_rows=rows.size();
    ret.pitch=pitch;
    return ret; 
  }

  typedef Scalar        ScalarType;
  typedef LocalOrdinal  LocalOrdinalType;
  typedef GlobalOrdinal GlobalOrdinalType;

  bool                       has_local_indices;
  std::vector<GlobalOrdinal> rows;
  std::vector<GlobalOrdinal> cols;
  std::vector<Scalar>        coefs;
  
  thrust::device_vector<GlobalOrdinal> d_rows;
  thrust::device_vector<GlobalOrdinal> d_cols;
  thrust::device_vector<Scalar>        d_coefs;

  LocalOrdinal               num_cols;
  LocalOrdinal               num_cols_per_row;
  GlobalOrdinal              pitch;

#ifdef HAVE_MPI
  std::vector<GlobalOrdinal> external_index;
  std::vector<GlobalOrdinal>  external_local_index;
  std::vector<GlobalOrdinal> elements_to_send;
  thrust::device_vector<GlobalOrdinal> d_elements_to_send;
  thrust::device_vector<int> d_external_map;
  std::vector<int>           neighbors;
  std::vector<LocalOrdinal>  recv_length;
  std::vector<LocalOrdinal>  send_length;
#ifndef GPUDIRECT
  std::vector<Scalar>        send_buffer;
#endif
  thrust::device_vector<Scalar> d_send_buffer;
  std::vector<MPI_Request>   request;
#endif

  size_t num_nonzeros() const
  {
    return rows.size()*num_cols_per_row;
  }

  void reserve_space(unsigned nrows, unsigned ncols_per_row)
  {
    //compute pitch so that columns always begin aligned
    pitch=(nrows+31)/32*32; 

    num_cols_per_row = ncols_per_row;
 
    nvtxRangeId_t r1=nvtxRangeStartA("allocate device memory");
    d_rows.resize(nrows);
    d_cols.resize(pitch * ncols_per_row);
    d_coefs.resize(pitch * ncols_per_row);
    nvtxRangeEnd(r1);
#ifdef HAVE_MPI
    d_external_map.resize(nrows);
#endif
   
#if 0
    //These have been moved after kernel launches so they can overlap with work on the device.
    nvtxRangeId_t r2=nvtxRangeStartA("allocate host memory");
    rows.resize(nrows);
    cols.resize(pitch * ncols_per_row);
    nvtxRangeEnd(r2);

    nvtxRangeId_t r3=nvtxRangeStartA("register host memory");
    cudaHostRegister(&cols[0],sizeof(GlobalOrdinalType)* pitch * ncols_per_row, 0);
    cudaCheckError();
    nvtxRangeEnd(r3);
#endif
  }

  LocalOrdinalType get_local_row(GlobalOrdinalType row) {
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
      typename std::vector<GlobalOrdinal>::iterator row_iter =
          std::lower_bound(rows.begin(), rows.end(), row);
  
      //if we still haven't found row, it's not local so jump out:
      if (row_iter == rows.end() || *row_iter != row) {
        return -1;
      }
  
      local_row = row_iter - rows.begin();
    }
    return local_row;
  }

  void get_row_pointers(GlobalOrdinalType row, size_t& row_length,
                        GlobalOrdinalType*& cols_ptr,
                        ScalarType*& coefs_ptr)
  {
    ptrdiff_t local_row = get_local_row(row);

    if(local_row<0) return;

    cols_ptr = &cols[local_row*num_cols_per_row];
    coefs_ptr = &coefs[local_row*num_cols_per_row];
    
    int idx = num_cols_per_row-1;
    while(idx>=0) {
      if (cols_ptr[idx] != 0) break;
      --idx;
    }
    row_length = idx+1;
  }
};

}//namespace miniFE

#endif

