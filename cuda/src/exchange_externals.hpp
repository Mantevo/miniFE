#ifndef _exchange_externals_hpp_
#define _exchange_externals_hpp_

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

#include <cstdlib>
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <outstream.hpp>

#include <TypeTraits.hpp>

namespace miniFE {

template<typename Scalar, typename Index> 
  __global__ void copyElementsToBuffer(Scalar *src, Scalar *dst, Index *indices, int N) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    int idx=indices[i];
    dst[i]=__ldg(src+idx);
  }
}

template<typename MatrixType,
         typename VectorType>
void
exchange_externals(MatrixType& A,
                   VectorType& x)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "entering exchange_externals\n";
#endif

  int numprocs = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if (numprocs < 2) return;
  
  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const std::vector<LocalOrdinal>& recv_length = A.recv_length;
  const std::vector<LocalOrdinal>& send_length = A.send_length;
  const std::vector<int>& neighbors = A.neighbors;
  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  std::vector<MPI_Request>& request = A.request;

  //
  // Externals are at end of locals
  //
  
  //
  // Fill up send buffer
  //

  int BLOCK_SIZE=256;
  int BLOCKS=min((int)(A.d_elements_to_send.size()+BLOCK_SIZE-1)/BLOCK_SIZE,2048);

  copyElementsToBuffer<<<BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(thrust::raw_pointer_cast(&x.d_coefs[0]),
                                     thrust::raw_pointer_cast(&A.d_send_buffer[0]), 
                                     thrust::raw_pointer_cast(&A.d_elements_to_send[0]),
                                     A.d_elements_to_send.size());
  cudaCheckError();

#ifndef GPUDIRECT
  std::vector<Scalar>& send_buffer = A.send_buffer;
  //wait for packing to finish
  cudaMemcpyAsync(&send_buffer[0],thrust::raw_pointer_cast(&A.d_send_buffer[0]),sizeof(Scalar)*A.d_elements_to_send.size(),cudaMemcpyDeviceToHost,CudaManager::s1);
  cudaCheckError();
#endif
  cudaEventRecord(CudaManager::e1,CudaManager::s1);

#ifdef GPUDIRECT
  Scalar * x_external = thrust::raw_pointer_cast(&x.d_coefs[local_nrow]);
#else
  std::vector<Scalar>& x_coefs = x.coefs;
  Scalar* x_external = &(x_coefs[local_nrow]);
#endif

  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

  // Post receives first
  for(int i=0; i<num_neighbors; ++i) {
    int n_recv = recv_length[i];
    MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, &request[i]);
    x_external += n_recv;
  }

#ifdef MINIFE_DEBUG
  os << "launched recvs\n";
#endif


  //
  // Send to each neighbor
  //

#ifdef GPUDIRECT
  Scalar* s_buffer = thrust::raw_pointer_cast(&A.d_send_buffer[0]);
#else
  Scalar* s_buffer = &send_buffer[0];
#endif
  //wait for packing or copy to host to finish
  cudaEventSynchronize(CudaManager::e1);
  cudaCheckError();

  for(int i=0; i<num_neighbors; ++i) {
    int n_send = send_length[i];
    MPI_Send(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD);
    s_buffer += n_send;
  }

#ifdef MINIFE_DEBUG
  os << "send to " << num_neighbors << std::endl;
#endif

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  for(int i=0; i<num_neighbors; ++i) {
    if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
  
#ifndef GPUDIRECT
  x.copyToDeviceAsync(local_nrow,CudaManager::s1);
#endif

#ifdef MINIFE_DEBUG
  os << "leaving exchange_externals"<<std::endl;
#endif

//endif HAVE_MPI
#endif
}

#ifdef HAVE_MPI
static std::vector<MPI_Request> exch_ext_requests;
#endif

template<typename MatrixType,
         typename VectorType>
void
begin_exchange_externals(MatrixType& A,
                         VectorType& x)
{
#ifdef HAVE_MPI

  int numprocs = 1, myproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  if (numprocs < 2) return;

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const std::vector<LocalOrdinal>& recv_length = A.recv_length;
  const std::vector<int>& neighbors = A.neighbors;

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  exch_ext_requests.resize(num_neighbors*2);

  //
  // Externals are at end of locals
  //
#ifdef GPUDIRECT
  Scalar * x_external = thrust::raw_pointer_cast(&x.d_coefs[local_nrow]);
#else
  std::vector<Scalar>& x_coefs = x.coefs;
  Scalar* x_external = &(x_coefs[local_nrow]);
#endif

  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

  // Post receives first
  for(int i=0; i<num_neighbors; ++i) {
    int n_recv = recv_length[i];
    MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, &exch_ext_requests[i]);
    x_external += n_recv;
  }

  //
  // Fill up send buffer
  //
  int BLOCK_SIZE=256;
  int BLOCKS=min((int)(A.d_elements_to_send.size()+BLOCK_SIZE-1)/BLOCK_SIZE,2048);

  cudaEventRecord(CudaManager::e1,CudaManager::s1);
  cudaStreamWaitEvent(CudaManager::s2,CudaManager::e1,0);

  copyElementsToBuffer<<<BLOCKS,BLOCK_SIZE,0,CudaManager::s2>>>(thrust::raw_pointer_cast(&x.d_coefs[0]),
                                     thrust::raw_pointer_cast(&A.d_send_buffer[0]), 
                                     thrust::raw_pointer_cast(&A.d_elements_to_send[0]),
                                     A.d_elements_to_send.size());
  cudaCheckError();
  //This isn't necessary for correctness but I want to make sure this starts before the interrior kernel
  cudaStreamWaitEvent(CudaManager::s1,CudaManager::e2,0); 
#ifndef GPUDIRECT
  std::vector<Scalar>& send_buffer = A.send_buffer;
  cudaMemcpyAsync(&send_buffer[0],thrust::raw_pointer_cast(&A.d_send_buffer[0]),sizeof(Scalar)*A.d_elements_to_send.size(),cudaMemcpyDeviceToHost,CudaManager::s2);
  cudaCheckError();
#endif
  cudaEventRecord(CudaManager::e2,CudaManager::s2);

#endif
}

template<typename MatrixType,
         typename VectorType>
inline
void
finish_exchange_externals(MatrixType &A, VectorType &x)
{
#ifdef HAVE_MPI
  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
  
  const std::vector<LocalOrdinal>& send_length = A.send_length;
  const std::vector<int>& neighbors = A.neighbors;
  int num_neighbors = A.neighbors.size();
  MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();
  int MPI_MY_TAG = 99;
  
  //
  // Send to each neighbor
  //

#ifdef GPUDIRECT
  Scalar* s_buffer = thrust::raw_pointer_cast(&A.d_send_buffer[0]);
#else
  Scalar* s_buffer = &A.send_buffer[0];
#endif
  
  //wait for packing or copy to host to finish
  cudaEventSynchronize(CudaManager::e2);
  cudaCheckError();

  for(int i=0; i<num_neighbors; ++i) {
    int n_send = send_length[i];
    MPI_Isend(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD, &exch_ext_requests[num_neighbors+i]);
    s_buffer += n_send;
  }
  //
  // Complete the reads issued above
  //

  MPI_Status status;
  for(int i=0; i<exch_ext_requests.size(); ++i) {
    if (MPI_Wait(&exch_ext_requests[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

//endif HAVE_MPI
#endif
}

}//namespace miniFE

#endif

