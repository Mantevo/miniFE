#ifndef _cg_solve_hpp_
#define _cg_solve_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia	Corporation
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
//
// ************************************************************************
//@HEADER

#include <cmath>
#include <limits>

#include <Vector_functions.hpp>
#include <mytimer.hpp>

#include <outstream.hpp>
#include "nvamg_c.h"

namespace miniFE {

template<typename Scalar>
void print_vec(const std::vector<Scalar>& vec, const std::string& name)
{
  for(size_t i=0; i<vec.size(); ++i) {
    std::cout << name << "["<<i<<"]: " << vec[i] << std::endl;
  }
}

template<typename VectorType>
bool breakdown(typename VectorType::ScalarType inner,
               const VectorType& v,
               const VectorType& w)
{
  typedef typename VectorType::ScalarType Scalar;
  typedef typename TypeTraits<Scalar>::magnitude_type magnitude;

//This is code that was copied from Aztec, and originally written
//by my hero, Ray Tuminaro.
//
//Assuming that inner = <v,w> (inner product of v and w),
//v and w are considered orthogonal if
//  |inner| < 100 * ||v||_2 * ||w||_2 * epsilon

  magnitude vnorm = std::sqrt(dot(v,v));
  magnitude wnorm = std::sqrt(dot(w,w));
  return std::abs(inner) <= 100*vnorm*wnorm*std::numeric_limits<magnitude>::epsilon();
}

template<typename OperatorType,
         typename VectorType,
         typename Matvec>
void
cg_solve(OperatorType& A,
         const VectorType& b,
         VectorType& x,
         Matvec matvec,
         typename OperatorType::LocalOrdinalType max_iter,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& tolerance,
         typename OperatorType::LocalOrdinalType& num_iters,
         typename TypeTraits<typename OperatorType::ScalarType>::magnitude_type& normr,
         timer_type* my_cg_times)
{
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename OperatorType::GlobalOrdinalType GlobalOrdinalType;
  typedef typename OperatorType::LocalOrdinalType LocalOrdinalType;
  typedef typename TypeTraits<ScalarType>::magnitude_type magnitude_type;

  timer_type t0 = 0, tWAXPY = 0, tDOT = 0, tMATVEC = 0, tMATVECDOT = 0;
  timer_type total_time = mytimer();

  int myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  if (!A.has_local_indices) {
    std::cerr << "miniFE::cg_solve ERROR, A.has_local_indices is false, needs to be true. This probably means "
       << "miniFE::make_local_matrix(A) was not called prior to calling miniFE::cg_solve."
       << std::endl;
    return;
  }

  char* str;
  int ngpu = 2;
  int local_rank = 0;
  int device = 0;
  int skip_gpu = 99999;
  if((str = getenv("CUDA_NGPU")) != NULL) {
    ngpu = atoi(str);
  }
  if((str = getenv("CUDA_SKIP_GPU")) != NULL) {
    skip_gpu = atoi(str);
  }
  if((str = getenv("SLURM_LOCALID")) != NULL) {
    local_rank = atoi(str);
    device = local_rank % ngpu;
    if(device >= skip_gpu) device++;
  }
  if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
    local_rank = atoi(str);
    device = local_rank % ngpu;
    if(device >= skip_gpu) device++;
  }
  if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
    local_rank = atoi(str);
    device = local_rank % ngpu;
    if(device >= skip_gpu) device++;
  }

  size_t nrows = A.rows.size();
  LocalOrdinalType ncols = A.num_cols;

  NVAMG_SAFE_CALL(NVAMG_initialize());
  NVAMG_SAFE_CALL(NVAMG_initialize_plugins());
  NVAMG_matrix_handle matrix;
  NVAMG_vector_handle rhs;
  NVAMG_vector_handle soln;
  NVAMG_resources_handle rsrc = NULL;
  NVAMG_solver_handle solver = NULL;
  NVAMG_config_handle config;
  NVAMG_SAFE_CALL(NVAMG_config_create_from_file(&config,"NVAMG_CONFIG" ));

  MPI_Comm nvamg_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &nvamg_comm);
  int devices[] = {device};

  NVAMG_resources_create(&rsrc, config, &nvamg_comm, 1, devices);
  NVAMG_SAFE_CALL(NVAMG_solver_create(&solver, rsrc, NVAMG_mode_dDDI, config));
  NVAMG_SAFE_CALL(NVAMG_matrix_create(&matrix, rsrc, NVAMG_mode_dDDI));
  NVAMG_SAFE_CALL(NVAMG_vector_create(&rhs, rsrc, NVAMG_mode_dDDI));
  NVAMG_SAFE_CALL(NVAMG_vector_create(&soln, rsrc, NVAMG_mode_dDDI));

  //Generating communication Maps for NVAMG
  if(A.neighbors.size()>0) {
    int** send_map = new int*[A.neighbors.size()];
    int** recv_map = new int*[A.neighbors.size()];
    int send_offset = 0;
    int recv_offset = A.row_offsets.size()-1;;
    for(int i = 0; i<A.neighbors.size();i++) {
      send_map[i] = &A.elements_to_send[send_offset];
      send_offset += A.send_length[i];
      recv_map[i] = new int[A.recv_length[i]];
      for(int j=0; j<A.recv_length[i]; j++)
        recv_map[i][j] = recv_offset+j;
      recv_offset += A.recv_length[i];
    }
    const int** send_map_c = (const int**) send_map;
    const int** recv_map_c = (const int**) recv_map;
    NVAMG_SAFE_CALL(NVAMG_matrix_comm_from_maps_one_ring(
      matrix, 1, A.neighbors.size(),A.neighbors.data(),
      A.send_length.data(), send_map_c,
      A.recv_length.data(), recv_map_c));
    NVAMG_SAFE_CALL(NVAMG_vector_bind(rhs,matrix));
    NVAMG_SAFE_CALL(NVAMG_vector_bind(soln,matrix));
    for(int i=0; i<A.neighbors.size(); i++)
      delete [] recv_map[i];

  }

  for(int i=0;i<x.coefs.size();i++) x.coefs[i]=1;

  VectorType r(b.startIndex, nrows);
  VectorType p(0, ncols);
  VectorType Ap(b.startIndex, nrows);

  normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;

  LocalOrdinalType print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;

  ScalarType one = 1.0;
  ScalarType zero = 0.0;

  TICK(); waxpby(one, x, zero, x, p); TOCK(tWAXPY);

  TICK();
  matvec(A, p, Ap);
  TOCK(tMATVEC);

  TICK(); waxpby(one, b, -one, Ap, r); TOCK(tWAXPY);

  TICK(); rtrans = dot_r2(r); TOCK(tDOT);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }
  {

    //Matrix upload needs to happen before vector, otherwise it crashes
    NVAMG_SAFE_CALL(NVAMG_matrix_upload_all(matrix,A.row_offsets.size()-1, A.packed_coefs.size(),1,1, &A.row_offsets[0],&A.packed_cols[0],&A.packed_coefs[0], NULL));
    NVAMG_SAFE_CALL(NVAMG_vector_upload(soln, p.coefs.size(), 1, &p.coefs[0]));
    NVAMG_SAFE_CALL(NVAMG_vector_upload(rhs, b.coefs.size(), 1, &b.coefs[0]));

    int n = 0;
    int bsize_x = 0, bsize_y = 0;

    NVAMG_SAFE_CALL(NVAMG_solver_setup(solver, matrix));
    NVAMG_SAFE_CALL(NVAMG_solver_solve(solver, rhs, soln));
    NVAMG_SAFE_CALL(NVAMG_vector_download(soln, &x.coefs[0]));

    int niter;
    NVAMG_SAFE_CALL(NVAMG_solver_get_iterations_number(solver, &niter));

    TICK(); waxpby(one, x, zero, x, p); TOCK(tWAXPY);
    TICK();
    matvec(A, p, Ap);
    TOCK(tMATVEC);

    TICK(); waxpby(one, b, -one, Ap, r); TOCK(tWAXPY);

    TICK(); rtrans = dot_r2(r); TOCK(tDOT);

    normr = std::sqrt(rtrans);

    if (myproc == 0) {
      std::cout << "Final Residual = "<< normr << " after " << niter << " iterations" << std::endl;
    }
   }

  my_cg_times[WAXPY] = tWAXPY;
  my_cg_times[DOT] = tDOT;
  my_cg_times[MATVEC] = tMATVEC;
  my_cg_times[MATVECDOT] = tMATVECDOT;
  my_cg_times[TOTAL] = mytimer() - total_time;
}

}//namespace miniFE

/*if(true) {
for(int i = 0;i<A.row_offsets.size()-1;i++) {
  for(int j=A.row_offsets[i];j<A.row_offsets[i+1];j++)
    std::cout << "MATRIXPRINT "<<i<<" "<<A.packed_cols[j]<<" "<<A.packed_coefs[j]<<std::endl;
}
for(int i = 0;i<A.num_cols;i++) {
  std::cout << "VECTORPRINT "<<i<<" "<<b.coefs[i]<<std::endl;
}
for(int i = 0;i<A.num_cols;i++) {
  std::cout << "VECTORXPRINT "<<i<<" "<<x.coefs[i]<<std::endl;
}
}*/
#endif

