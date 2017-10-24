#ifndef _SparseMatrix_functions_hpp_
#define _SparseMatrix_functions_hpp_

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
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <Vector.hpp>
#include <Vector_functions.hpp>
#include <ElemData.hpp>
#include <exchange_externals.hpp>
#include <mytimer.hpp>

#ifdef MINIFE_HAVE_TBB
#include <LockingMatrix.hpp>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <assert.h>

namespace miniFE {

  /**************************************************
   *  * structs for converting between signed and unsigned values without 
   *   * type casting.
   *    * ************************************************/
  /*****************************
   *  * Generic converter for unsigned types.
   *   * This becomes a no op
   *    *****************************/
  template <class GlobalOrdinal>
    struct intuint {
      union {
        GlobalOrdinal ival;
        GlobalOrdinal uval;
      };
    };
  /***************************
   *  * char converter
   *   **************************/
  template <>
    struct intuint<char> {
      union {
        char ival;
        unsigned char uval;
      };
    };

  /***************************
   *  * Short converter
   *   **************************/
  template <>
    struct intuint<short> {
      union {
        short ival;
        unsigned short uval;
      };
    };

  /***************************
   *  * Integer converter
   *   **************************/
  template <>
    struct intuint<int> {
      union {
        int ival;
        unsigned int uval;
      };
    };

  /***************************
   *  * long converter
   *   **************************/
  template <>
    struct intuint<long> {
      union {
        long ival;
        unsigned long uval;
      };
    };


  template<typename MatrixType>
    void write_matrix(const std::string& filename, 
        MatrixType& mat)
    {
      typedef typename MatrixType::LocalOrdinalType LocalOrdinalType;
      typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;
      typedef typename MatrixType::ScalarType ScalarType;

      mat.copyToHost();

      int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

      std::ostringstream osstr;
      osstr << filename << "." << numprocs << "." << myproc;
      std::string full_name = osstr.str();
      std::ofstream ofs(full_name.c_str());
      //ofs << std::setprecision(16);

      size_t nrows = mat.rows.size();
      size_t nnz = mat.num_nonzeros();

      for(int p=0; p<numprocs; ++p) {
        if (p == myproc) {
          if (p == 0) {
            ofs << nrows << " " << nnz << std::endl;
          }
          for(size_t i=0; i<nrows; ++i) {
            int offset=i;

            for(int j=0;j<mat.num_cols_per_row;j++) {
              if(mat.cols[offset]!=-1)
                ofs << mat.rows[i] << " " << mat.cols[offset] << " " << mat.coefs[offset] << std::endl;
              offset+=mat.pitch;
            }
          }
        }
#ifdef HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
      }
    }

  /****************************************************************
   * This version is pitched so that memory accesses will be colesced 
   * across threads
   ****************************************************************/
  template <class GlobalOrdinal>
    __device__  int inline pitchedBinarySearch(GlobalOrdinal *indices, GlobalOrdinal low, GlobalOrdinal high, GlobalOrdinal _val, const GlobalOrdinal pitch)
    {
      GlobalOrdinal retval=-1;

      intuint<GlobalOrdinal> val;
      val.ival=_val;

      while(high>=low)
      {
        GlobalOrdinal mid=low+(high-low)/2;
        intuint<GlobalOrdinal> mval;
        //mval.ival=indices[pitch*mid];
        mval.ival=__ldg(indices+pitch*mid);
        if(mval.uval>val.uval)
          high=mid-1;
        else if (mval.uval<val.uval)
          low=mid+1;
        else
        {
          retval=mid;
          break;
        }
      }
      return retval;
    }


  template<typename MatrixType, typename VectorType>
    __device__ __inline__
    void 
    sum_into_global_linear_system_cuda(typename MatrixType::GlobalOrdinalType elem_node_ids[Hex8::numNodesPerElem], 
        typename MatrixType::ScalarType elem_diffusion_matrix[Hex8::numNodesPerElem*Hex8::numNodesPerElem],
        typename MatrixType::ScalarType elem_source_vector[Hex8::numNodesPerElem], 
        MatrixType A, VectorType b)
    {
      typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
      typedef typename MatrixType::ScalarType Scalar;

#pragma unroll
      for(int elethidx=0; elethidx<Hex8::numNodesPerElem; ++elethidx)
      {
        GlobalOrdinal row = elem_node_ids[elethidx];

        int local_row = -1;
        //first see if we can get the local-row index using fast direct lookup:
        int idx= row - A.rows[0];
        if(idx>=0 && idx < A.num_rows && A.rows[idx] == row)
          local_row = idx;
        else {
          //if we didn't get the local-row index using direct lookup, try a
          //more expensive binary-search:
          local_row=binarySearch(A.rows, 0, A.num_rows-1, row);
          if(local_row==-1)
            continue;
        }

        GlobalOrdinal* mat_row_cols = &A.cols[local_row];
        Scalar* mat_row_coefs = &A.coefs[local_row];

#pragma unroll
        for(size_t i=0; i<Hex8::numNodesPerElem; ++i) {
          //find the location to apply the coef
          GlobalOrdinal loc=pitchedBinarySearch<GlobalOrdinal>(mat_row_cols,0,A.num_cols_per_row-1,elem_node_ids[i],A.pitch);

          if (loc!=-1) {
            Scalar coef;
            if(i<elethidx)
              coef = elem_diffusion_matrix[i*(2*Hex8::numNodesPerElem-i+1)/2+elethidx-i];
            else
              coef = elem_diffusion_matrix[elethidx*(2*Hex8::numNodesPerElem-elethidx+1)/2+i-elethidx];

            miniFEAtomicAdd(&mat_row_coefs[A.pitch*loc], coef);
            //mat_row_coefs[A.pitch*loc]+=coef;
          }
        }
      }

      sum_into_vector_cuda(Hex8::numNodesPerElem, elem_node_ids, elem_source_vector, b);
    }

  template<typename MatrixType>
    double
    parallel_memory_overhead_MB(const MatrixType& A)
    {
      typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
      typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
      double mem_MB = 0;

#ifdef HAVE_MPI
      double invMB = 1.0/(1024*1024);
      mem_MB = invMB*A.external_index.size()*sizeof(GlobalOrdinal);
      mem_MB += invMB*A.external_local_index.size()*sizeof(GlobalOrdinal);
      mem_MB += invMB*A.elements_to_send.size()*sizeof(GlobalOrdinal);
      mem_MB += invMB*A.neighbors.size()*sizeof(int);
      mem_MB += invMB*A.recv_length.size()*sizeof(LocalOrdinal);
      mem_MB += invMB*A.send_length.size()*sizeof(LocalOrdinal);

      double tmp = mem_MB;
      MPI_Allreduce(&tmp, &mem_MB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

      return mem_MB;
    }

  template<typename MatrixType, typename VectorType>
    __global__ void
    impose_dirichlet_first_kernel(typename MatrixType::ScalarType prescribed_value,
        MatrixType A,
        VectorType b,
        typename MatrixType::GlobalOrdinalType *bc_rows,
        typename MatrixType::GlobalOrdinalType num_bc_rows,
        typename MatrixType::GlobalOrdinalType first_local_row,
        typename MatrixType::GlobalOrdinalType last_local_row
        ) {
      typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

      for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<num_bc_rows;i+=blockDim.x*gridDim.x) {
        GlobalOrdinal row = bc_rows[i];
        //If row is in my local range
        if(row >= first_local_row && row <= last_local_row) {
          GlobalOrdinal local_row = row-first_local_row;
          //set prescribed value
          b.coefs[local_row] = prescribed_value;
          //zero columns and add 1 to the diagonal
          for(int j=0; j<A.num_cols_per_row ; j++) {
            GlobalOrdinal idx=local_row+j*A.pitch;
            GlobalOrdinal col=A.cols[idx];
            if(col==-1) break;
            A.coefs[idx]=  (col==row)?1:0; 
          }
        }
      }
    }

  template<typename MatrixType, typename VectorType>
    __global__ void
    impose_dirichlet_second_kernel(typename MatrixType::ScalarType prescribed_value,
        MatrixType A,
        VectorType b,
        typename MatrixType::GlobalOrdinalType *bc_rows,
        typename MatrixType::GlobalOrdinalType num_bc_rows
        ) {
      typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
      typedef typename MatrixType::ScalarType Scalar;

      for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<A.num_rows;i+=blockDim.x*gridDim.x) {
        GlobalOrdinal local_row = i;
        GlobalOrdinal row = A.rows[i];
        //if this row is a bc_row
        if (binarySearch(bc_rows,0,num_bc_rows-1,row)!=-1) continue;
        Scalar sum=0;

        //zero row 
        for(int j=0; j<A.num_cols_per_row; j++) {
          int idx=local_row+j*A.pitch;
          GlobalOrdinal col=A.cols[idx];

          if(col==-1) break;

          if(binarySearch(bc_rows,0,num_bc_rows-1,col)!=-1) {
            sum+=A.coefs[idx];
            A.coefs[idx]=0;
          }
        }
        //add deleted entries to the RHS
        b.coefs[i] -= sum*prescribed_value;
      }
    }

  template<typename MatrixType,
    typename VectorType>
      void
      impose_dirichlet(typename MatrixType::ScalarType prescribed_value,
          MatrixType& A,
          VectorType& b,
          int global_nx,
          int global_ny,
          int global_nz,
          const std::set<typename MatrixType::GlobalOrdinalType>& bc_rows)
      {
        typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
        typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
        typedef typename MatrixType::ScalarType Scalar;

        GlobalOrdinal first_local_row = A.rows.size()>0 ? A.rows[0] : 0;
        GlobalOrdinal last_local_row  = A.rows.size()>0 ? A.rows[A.rows.size()-1] : -1;
        
        int num_bc_rows=bc_rows.size();
        int num_rows=A.rows.size();

        if(num_bc_rows>0) {
          std::vector<GlobalOrdinal> bc_rows_vec(bc_rows.begin(),bc_rows.end());
          thrust::device_vector<GlobalOrdinal> d_bc_rows_vec(num_bc_rows);
          cudaMemcpyAsync(thrust::raw_pointer_cast(&d_bc_rows_vec[0]),&bc_rows_vec[0],sizeof(GlobalOrdinal)*num_bc_rows,cudaMemcpyHostToDevice,CudaManager::s1);

          int BLOCK_SIZE=256;
          int MAX_BLOCKS=8192;
          int NUM_BLOCKS;

          NUM_BLOCKS=min(MAX_BLOCKS,(int)(num_bc_rows+BLOCK_SIZE-1)/BLOCK_SIZE);
          //TODO place in different streams...
          impose_dirichlet_first_kernel<<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(prescribed_value, A.getPOD(), b.getPOD(), thrust::raw_pointer_cast(&d_bc_rows_vec[0]),num_bc_rows,first_local_row,last_local_row);
          cudaCheckError();
          NUM_BLOCKS=min(MAX_BLOCKS,(int)(num_rows+BLOCK_SIZE-1)/BLOCK_SIZE);
          impose_dirichlet_second_kernel<<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(prescribed_value, A.getPOD(), b.getPOD(), thrust::raw_pointer_cast(&d_bc_rows_vec[0]),num_bc_rows);
          cudaCheckError();
        }
      }

  static timer_type exchtime = 0;
  
  enum MATVEC_RANGE { INTERNAL=0, EXTERNAL=1 };
  __device__ __inline__ MATVEC_RANGE get_range(int mask, int bit) {
      return static_cast<MATVEC_RANGE>( (mask>>bit)&1);
  }

  template<typename MatrixType>
      __global__ void createExternalMapping(MatrixType A) {
        typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;
        for(int row_idx=blockIdx.x*blockDim.x+threadIdx.x;row_idx<A.num_rows;row_idx+=blockDim.x*gridDim.x)
        {
          int offset=row_idx;
          int mask=1;
          int bitfield=0;
          for(int j=0;j<A.num_cols_per_row;++j) {
            GlobalOrdinalType col=A.cols[offset];
          
            //if this column is larger than the number of rows in A it is an external
            if(col!=-1 && col>=A.num_rows)
              bitfield|=mask;
            mask<<=1;
            offset+=A.pitch;
          }
          A.external_map[row_idx]=bitfield;
        }
      }

  //TODO Kahan Summantion?
  //------------------------------------------------------------------------
  //Compute matrix vector product y = A*x where:
  //
  // A - input matrix
  // x - input vector
  // y - result vector
  //
  template<typename MatrixType,
    typename VectorType>
      __global__ void matvec_ell_kernel(MatrixType A, VectorType X, VectorType Y) {
        typedef typename MatrixType::ScalarType ScalarType;
        typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;

        for(int row_idx=blockIdx.x*blockDim.x+threadIdx.x;row_idx<A.num_rows;row_idx+=blockDim.x*gridDim.x)
        {
          ScalarType sum=0;
          GlobalOrdinalType offset = row_idx;
          for(int j=0;j<A.num_cols_per_row;++j)
          {
            GlobalOrdinalType c=A.cols[offset];
            if(c!=-1) {
              ScalarType a=A.coefs[offset]; 
              ScalarType x=__ldg(X.coefs+c);
              sum+=a*x;
            }
            offset+=A.pitch;
          }
          Y.coefs[row_idx]=sum;
        }
      }

  template<MATVEC_RANGE RANGE, typename MatrixType,
    typename VectorType>
      __global__ void matvec_overlap_ell_kernel(MatrixType A, VectorType X, VectorType Y) {
        typedef typename MatrixType::ScalarType ScalarType;
        typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;

        for(int row_idx=blockIdx.x*blockDim.x+threadIdx.x;row_idx<A.num_rows;row_idx+=blockDim.x*gridDim.x)
        {
          ScalarType sum=0;
          GlobalOrdinalType offset = row_idx;
          int bitfield=A.external_map[row_idx];
          //quickly skip cases where there are no external entries
          if( RANGE==EXTERNAL && (bitfield==0) ) 
            continue;

          for(int j=0;j<A.num_cols_per_row;++j)
          {
            if(RANGE == get_range(bitfield,j)) {
              GlobalOrdinalType c=A.cols[offset];
              if(c!=-1) {
                ScalarType a=A.coefs[offset];
                ScalarType x=__ldg(X.coefs+c);
                sum+=a*x;
              }
            }
            offset+=A.pitch;
          }
          if(RANGE==INTERNAL)
            Y.coefs[row_idx]=sum;
          else
            Y.coefs[row_idx]+=sum;

        }
      }

  template<typename MatrixType,
    typename VectorType>
      struct matvec_std {
        void operator()(MatrixType& A,
            VectorType& x,
            VectorType& y)
        {
          typedef typename MatrixType::ScalarType ScalarType;
          typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;
          typedef typename MatrixType::LocalOrdinalType LocalOrdinalType;
          
          const int BLOCK_SIZE=256;
          const int MAX_BLOCKS=32768;
          int NUM_BLOCKS=min(MAX_BLOCKS,(int)(A.rows.size()+BLOCK_SIZE-1)/BLOCK_SIZE);

#ifndef MATVEC_OVERLAP
          exchange_externals(A, x);
          matvec_ell_kernel<<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(A.getPOD(), x.getPOD(), y.getPOD());
#else
          nvtxRangeId_t r1=nvtxRangeStartA("begin exchange");
          begin_exchange_externals(A,x);
          nvtxRangeEnd(r1);
          nvtxRangeId_t r2=nvtxRangeStartA("interier region");
          matvec_overlap_ell_kernel<INTERNAL><<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(A.getPOD(), x.getPOD(), y.getPOD());
          nvtxRangeEnd(r2);
          nvtxRangeId_t r3=nvtxRangeStartA("end exchange");
          finish_exchange_externals(A,x);
          nvtxRangeEnd(r3);
          nvtxRangeId_t r4=nvtxRangeStartA("exterier region");
          matvec_overlap_ell_kernel<EXTERNAL><<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(A.getPOD(), x.getPOD(), y.getPOD());
          nvtxRangeEnd(r4);
#endif
        }
      };

}//namespace miniFE

#endif

