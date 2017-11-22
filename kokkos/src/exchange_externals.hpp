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
#include <Kokkos_Vector.hpp>
namespace miniFE {

template<typename MatrixType,
         typename VectorType>
struct
exchange_externals_functor {
	typedef device_device_type device_type;
	Kokkos::vector<typename VectorType::ScalarType,Kokkos::DefaultExecutionSpace> _x;
    Kokkos::vector<typename MatrixType::GlobalOrdinalType,Kokkos::DefaultExecutionSpace> _elements_to_send;
	Kokkos::vector<typename MatrixType::ScalarType,Kokkos::DefaultExecutionSpace>        _send_buffer;

	exchange_externals_functor(const MatrixType& A,
			const VectorType& x):_x(x.coefs),_elements_to_send(A.elements_to_send),_send_buffer(A.send_buffer) {

	}
	KOKKOS_INLINE_FUNCTION
	void operator() (int i) const {
		_send_buffer.d_view(i) = _x.d_view(_elements_to_send.d_view(i));
	}

};

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

#ifndef GPU_MPI
  x.coefs.device_to_host();
#endif

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  // Extract Matrix pieces

  int local_nrow = A.rows.size();
  int num_neighbors = A.neighbors.size();
  const Kokkos::vector<LocalOrdinal,Kokkos::DefaultExecutionSpace>& recv_length = A.recv_length;
  const Kokkos::vector<LocalOrdinal,Kokkos::DefaultExecutionSpace>& send_length = A.send_length;
  const Kokkos::vector<int,Kokkos::DefaultExecutionSpace>& neighbors = A.neighbors;
  const Kokkos::vector<GlobalOrdinal,Kokkos::DefaultExecutionSpace>& elements_to_send = A.elements_to_send;

  Kokkos::vector<Scalar,Kokkos::DefaultExecutionSpace>& send_buffer = A.send_buffer;

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

  Kokkos::vector<Scalar,Kokkos::DefaultExecutionSpace>& x_coefs = x.coefs;
#ifndef GPU_MPI
  Scalar* x_external = x_coefs.h_view.ptr_on_device() + local_nrow;
#else
  Scalar* x_external = x_coefs.d_view.ptr_on_device() + local_nrow;
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
  // Fill up send buffer
  //

  size_t total_to_be_sent = elements_to_send.size();
#ifdef MINIFE_DEBUG
  os << "total_to_be_sent: " << total_to_be_sent << std::endl;
#endif

 /* for(size_t i=0; i<total_to_be_sent; ++i) {
#ifdef MINIFE_DEBUG
    //expensive index range-check:
    if (elements_to_send[i] < 0 || elements_to_send[i] > x.coefs.size()) {
      os << "error, out-of-range. x.coefs.size()=="<<x.coefs.size()<<", elements_to_send[i]=="<<elements_to_send[i]<<std::endl;
    }
#endif
    send_buffer[i] = x.coefs[elements_to_send[i]];
  }*/
  exchange_externals_functor<MatrixType,VectorType> f(A,x);
  Kokkos::parallel_for("exchange_externals",total_to_be_sent,f);

  //
  // Send to each neighbor
  //

#ifndef GPU_MPI
  send_buffer.device_to_host();
  Scalar* s_buffer = send_buffer.h_view.ptr_on_device();
#else
  Scalar* s_buffer = send_buffer.d_view.ptr_on_device();
#endif

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
#ifndef GPU_MPI
  x.coefs.host_to_device();
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
  const Kokkos::vector<LocalOrdinal,Kokkos::DefaultExecutionSpace>& recv_length = A.recv_length;
  const Kokkos::vector<LocalOrdinal,Kokkos::DefaultExecutionSpace>& send_length = A.send_length;
  const Kokkos::vector<int,Kokkos::DefaultExecutionSpace>& neighbors = A.neighbors;
  const Kokkos::vector<GlobalOrdinal,Kokkos::DefaultExecutionSpace>& elements_to_send = A.elements_to_send;

  Kokkos::vector<Scalar,Kokkos::DefaultExecutionSpace> send_buffer(elements_to_send.size(), 0);

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  exch_ext_requests.resize(num_neighbors);

  //
  // Externals are at end of locals
  //

  Kokkos::vector<Scalar,Kokkos::DefaultExecutionSpace>& x_coefs = x.coefs;
  Scalar* x_external = &(x_coefs[local_nrow]);

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

  size_t total_to_be_sent = elements_to_send.size();
  for(size_t i=0; i<total_to_be_sent; ++i) send_buffer[i] = x.coefs[elements_to_send[i]];

  //
  // Send to each neighbor
  //

  Scalar* s_buffer = &send_buffer[0];

  for(int i=0; i<num_neighbors; ++i) {
    int n_send = send_length[i];
    MPI_Send(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD);
    s_buffer += n_send;
  }
#endif
}

inline
void
finish_exchange_externals(int num_neighbors)
{
#ifdef HAVE_MPI
  //
  // Complete the reads issued above
  //

  MPI_Status status;
  for(int i=0; i<num_neighbors; ++i) {
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

