#ifndef _generate_matrix_structure_hpp_
#define _generate_matrix_structure_hpp_

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

#include <sstream>
#include <stdexcept>
#include <map>
#include <algorithm>

#include <simple_mesh_description.hpp>
#include <SparseMatrix_functions.hpp>
#include <box_utils.hpp>
#include <utils.hpp>
#include <Kokkos_Vector.hpp>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace miniFE {


template<typename MatrixType, class Device>
struct generate_matrix_structure_functor {
	typedef Device device_type;

    typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
	typedef typename MatrixType::LocalOrdinalType LocalOrdinal;

	const simple_mesh_description<typename MatrixType::GlobalOrdinalType> mesh;
	MatrixType* A;

	Kokkos::vector<GlobalOrdinal> rows;
	Kokkos::vector<LocalOrdinal> row_offsets;
	Kokkos::vector<int> row_coords;

	Box box;
	unsigned int box_dims[3];
	GlobalOrdinal global_nrows;
	int global_nodes_x, global_nodes_y, global_nodes_z;
	GlobalOrdinal nrows;

	generate_matrix_structure_functor(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& _mesh,
                          MatrixType* _A): mesh(_mesh),A(_A) {

		  global_nodes_x = mesh.global_box[0][1]+1;
		  global_nodes_y = mesh.global_box[1][1]+1;
		  global_nodes_z = mesh.global_box[2][1]+1;
		  copy_box(mesh.local_box, box);

		  //num-owned-nodes in each dimension is num-elems+1
		  //only if num-elems > 0 in that dimension *and*
		  //we are at the high end of the global range in that dimension:
		  if (box[0][1] > box[0][0] && box[0][1] == mesh.global_box[0][1]) ++box[0][1];
		  if (box[1][1] > box[1][0] && box[1][1] == mesh.global_box[1][1]) ++box[1][1];
		  if (box[2][1] > box[2][0] && box[2][1] == mesh.global_box[2][1]) ++box[2][1];

		  global_nrows = global_nodes_x;
		  global_nrows *= global_nodes_y*global_nodes_z;

		  nrows = get_num_ids<GlobalOrdinal>(box);
		  try {
		    A->reserve_space(nrows, 27);
		  }
		  catch(std::exception& exc) {
		    std::ostringstream osstr;
		    osstr << "One of A.rows.resize, A.row_offsets.resize, A.packed_cols.reserve or A.packed_coefs.reserve: nrows=" <<nrows<<": ";
		    osstr << exc.what();
		    std::string str1 = osstr.str();
		    throw std::runtime_error(str1);
		  }
		  rows = Kokkos::vector<GlobalOrdinal>(nrows,0);
		  row_offsets = Kokkos::vector<LocalOrdinal>(nrows+1,0);
		  row_coords = Kokkos::vector<int>(nrows*3,0);

		  box_dims[0] = box[0][1] - box[0][0];
		  box_dims[1] = box[1][1] - box[1][0];
		  box_dims[2] = box[2][1] - box[2][0];
	}

	inline
	void operator() (const int &roffset) const{
		  int iz = roffset/(box_dims[1]*box_dims[0]) + box[2][0];
		  int iy = (roffset/box_dims[0])%box_dims[1] + box[1][0];
		  int ix = roffset%box_dims[0] + box[0][0];

	      GlobalOrdinal row_id =
	          get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z,
	                                ix, iy, iz);
	      rows[roffset] = mesh.map_id_to_row(row_id);
	      row_coords[roffset*3] = ix;
	      row_coords[roffset*3+1] = iy;
	      row_coords[roffset*3+2] = iz;

	      int nnz  = 0;
	      for(int sz=-1; sz<=1; ++sz) {
	       for(int sy=-1; sy<=1; ++sy) {
	        for(int sx=-1; sx<=1; ++sx) {
	          GlobalOrdinal col_id =
	              get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z,
	                                   ix+sx, iy+sy, iz+sz);
	          if (col_id >= 0 && col_id < global_nrows) {
	            ++nnz;
	          }
	        }
	       }
	      }
	      row_offsets[roffset+1] = nnz;
	}
};

template<typename MatrixType>
int
generate_matrix_structure(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh,
                          MatrixType& A)
{
  int myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  int threw_exc = 0;
  try {
  struct generate_matrix_structure_functor<MatrixType,host_device_type> functor(mesh,&A);

  Kokkos::parallel_for("generate_matrix_structure<Host>",functor.box_dims[0]*functor.box_dims[1]*functor.box_dims[2],functor);
  host_device_type::fence();

  for(int i=0;i<functor.row_offsets.size()-1;i++) {
	functor.row_offsets[i+1]+=functor.row_offsets[i];
  }

  init_matrix(A, functor.rows, functor.row_offsets, functor.row_coords,
		      functor.global_nodes_x, functor.global_nodes_y, functor.global_nodes_z, functor.global_nrows, functor.mesh);
  }
  catch(...) {
    std::cout << "proc " << myproc << " threw an exception in generate_matrix_structure, probably due to running out of memory." << std::endl;
    threw_exc = 1;
  }
#ifdef HAVE_MPI
  int global_throw = 0;
  MPI_Allreduce(&threw_exc, &global_throw, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  threw_exc = global_throw;
#endif
  if (threw_exc) {
    return 1;
  }

  return 0;
}

}//namespace miniFE

#endif

