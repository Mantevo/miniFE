#ifndef _Vector_functions_hpp_
#define _Vector_functions_hpp_

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
#include <sstream>
#include <fstream>
#include <iomanip>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>
#include <CudaUtils.h>
namespace miniFE {

template<typename VectorType>
void write_vector(const std::string& filename,
                  const VectorType& vec)
{
  vec.copyToHost();
  int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  std::ostringstream osstr;
  osstr << filename << "." << numprocs << "." << myproc;
  std::string full_name = osstr.str();
  std::ofstream ofs(full_name.c_str());

  typedef typename VectorType::ScalarType ScalarType;

  const std::vector<ScalarType>& coefs = vec.coefs;
  for(int p=0; p<numprocs; ++p) {
    if (p == myproc) {
      if (p == 0) {
        ofs << vec.local_size << std::endl;
      }
  
      typename VectorType::GlobalOrdinalType first = vec.startIndex;
      for(size_t i=0; i<vec.local_size; ++i) {
        ofs << first+i << " " << coefs[i] << std::endl;
      }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

template<typename VectorType>
__device__  __inline__
void sum_into_vector_cuda(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* __restrict__ indices,
                     const typename VectorType::ScalarType* __restrict__ coefs,
                     VectorType& vec)
{

 typename VectorType::GlobalOrdinalType first = vec.startIndex;
#pragma unroll
  for(size_t i=0; i<num_indices; ++i) {
    size_t idx = indices[i] - first;

    if (idx >= vec.n) continue;

    miniFEAtomicAdd(&vec.coefs[idx], coefs[i]);
    //vec[idx] += coefs[i];
  }
}


template<typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* indices,
                     const typename VectorType::ScalarType* coefs,
                     VectorType& vec)
{
  typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
  typedef typename VectorType::ScalarType Scalar;

  GlobalOrdinal first = vec.startIndex;
  GlobalOrdinal last = first + vec.local_size - 1;

  std::vector<Scalar>& vec_coefs = vec.coefs;

  for(size_t i=0; i<num_indices; ++i) {
    if (indices[i] < first || indices[i] > last) continue;
    size_t idx = indices[i] - first;
    vec_coefs[idx] += coefs[i];
  }
}

//------------------------------------------------------------
//Compute the update of a vector with the sum of two scaled vectors where:
//
// w = alpha*x + beta*y
//
// x,y - input vectors
//
// alpha,beta - scalars applied to x and y respectively
//
// w - output vector
//
template <typename VectorType> 
__global__  void waxpby_kernel(typename VectorType::ScalarType alpha, const VectorType x, 
                               typename VectorType::ScalarType beta, const VectorType y, 
                               VectorType w) {
  
  for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<x.n;idx+=gridDim.x*blockDim.x)
  {
      w.coefs[idx] = alpha * x.coefs[idx] + beta * y.coefs[idx];
  }
}

template<typename VectorType>
void
  waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
         typename VectorType::ScalarType beta, const VectorType& y,
         VectorType& w)
{
  typedef typename VectorType::ScalarType ScalarType;
  
#ifdef MINIFE_DEBUG
  if (y.local_size < x.local_size || w.local_size < x.local_size) {
    std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x." << std::endl;
    return;
  }
#endif
  int n = x.coefs.size();
  int BLOCK_SIZE=256;
  int BLOCKS=min((n+BLOCK_SIZE-1)/BLOCK_SIZE,2048*16);

  waxpby_kernel<<<BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(alpha, x.getPOD(), beta, y.getPOD(), w.getPOD());
  cudaCheckError();
}

//-----------------------------------------------------------
//Compute the dot product of two vectors where:
//
// x,y - input vectors
//
// result - return-value
//
template<typename Vector>
__global__ void dot_kernel(const Vector x, const Vector y, typename TypeTraits<typename Vector::ScalarType>::magnitude_type *d) {

  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;
  const int BLOCK_SIZE=512; 

  magnitude sum=0;
  for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<x.n;idx+=gridDim.x*blockDim.x) {
    sum+=x.coefs[idx]*y.coefs[idx];
  }

  //Do a shared memory reduction on the dot product
  __shared__ volatile magnitude red[BLOCK_SIZE];
  red[threadIdx.x]=sum; 
  //__syncthreads(); if(threadIdx.x<512) {sum+=red[threadIdx.x+512]; red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<256)  {sum+=red[threadIdx.x+256]; red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<128)  {sum+=red[threadIdx.x+128]; red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<64)   {sum+=red[threadIdx.x+64];  red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<32)   {sum+=red[threadIdx.x+32];  red[threadIdx.x]=sum;}
  //the remaining ones don't need syncthreads because they are warp synchronous
                   if(threadIdx.x<16)   {sum+=red[threadIdx.x+16];  red[threadIdx.x]=sum;}  
                   if(threadIdx.x<8)    {sum+=red[threadIdx.x+8];   red[threadIdx.x]=sum;}  
                   if(threadIdx.x<4)    {sum+=red[threadIdx.x+4];   red[threadIdx.x]=sum;}  
                   if(threadIdx.x<2)    {sum+=red[threadIdx.x+2];   red[threadIdx.x]=sum;}
                   if(threadIdx.x<1)    {sum+=red[threadIdx.x+1];}

  //save partial dot products
  if(threadIdx.x==0) d[blockIdx.x]=sum;
}

template<typename Scalar>
__global__ void dot_final_reduce_kernel(Scalar *d) {
  const int BLOCK_SIZE=1024;
  Scalar sum=d[threadIdx.x];
  __shared__ volatile Scalar red[BLOCK_SIZE];
  
  red[threadIdx.x]=sum; 
  __syncthreads(); if(threadIdx.x<512)  {sum+=red[threadIdx.x+512]; red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<256)  {sum+=red[threadIdx.x+256]; red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<128)  {sum+=red[threadIdx.x+128]; red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<64)   {sum+=red[threadIdx.x+64];  red[threadIdx.x]=sum;} 
  __syncthreads(); if(threadIdx.x<32)   {sum+=red[threadIdx.x+32];  red[threadIdx.x]=sum;}
  //the remaining ones don't need syncthreads because they are warp synchronous
                   if(threadIdx.x<16)   {sum+=red[threadIdx.x+16];  red[threadIdx.x]=sum;}  
                   if(threadIdx.x<8)    {sum+=red[threadIdx.x+8];   red[threadIdx.x]=sum;}  
                   if(threadIdx.x<4)    {sum+=red[threadIdx.x+4];   red[threadIdx.x]=sum;}  
                   if(threadIdx.x<2)    {sum+=red[threadIdx.x+2];   red[threadIdx.x]=sum;}
                   if(threadIdx.x<1)    {sum+=red[threadIdx.x+1];}

  //save final dot product at the front
                   if(threadIdx.x==0) d[0]=sum;
}

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
  dot(const Vector& x,
      const Vector& y)
{
  typedef typename Vector::ScalarType Scalar;
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;

  int n = x.coefs.size();

#ifdef MINIFE_DEBUG
  if (y.local_size < n) {
    std::cerr << "miniFE::dot ERROR, y must be at least as long as x."<<std::endl;
    n = y.local_size;
  }
#endif

 int BLOCK_SIZE=512;
 int NUM_BLOCKS=min(1024,(n+BLOCK_SIZE-1)/BLOCK_SIZE);
 static thrust::device_vector<magnitude> d(1024);
 cudaMemset_custom(thrust::raw_pointer_cast(&d[0]),(magnitude)0,1024,CudaManager::s1);

 dot_kernel<<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(x.getPOD(), y.getPOD(), thrust::raw_pointer_cast(&d[0]));
 cudaCheckError();
 dot_final_reduce_kernel<<<1,1024,0,CudaManager::s1>>>(thrust::raw_pointer_cast(&d[0]));
 cudaCheckError();
 
 static magnitude result;

 //TODO move outside?
 static bool first=true;
 if(first==true) {
   cudaHostRegister(&result,sizeof(result),0);
   first=false;
 }

 //TODO do this with GPU direct?
 cudaMemcpyAsync(&result,thrust::raw_pointer_cast(&d[0]),sizeof(magnitude),cudaMemcpyDeviceToHost,CudaManager::s1);
 cudaEventRecord(CudaManager::e1,CudaManager::s1);
 cudaEventSynchronize(CudaManager::e1);

#ifdef HAVE_MPI
  nvtxRangeId_t r1=nvtxRangeStartA("MPI All Reduce");
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
  nvtxRangeEnd(r1);
  return global_dot;
#else
  return result;
#endif
}

}//namespace miniFE

#endif

