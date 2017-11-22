#ifndef _perform_element_loop_hpp_
#define _perform_element_loop_hpp_

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

#include <BoxIterator.hpp>
#include <simple_mesh_description.hpp>
#include <SparseMatrix_functions.hpp>
#include <box_utils.hpp>
#include <Hex8_box_utils.hpp>
#include <Hex8_ElemData.hpp>
#include <cstdio>
#include <Kokkos_Types.hpp>
namespace miniFE {

template<typename GlobalOrdinal,
         typename MatrixType, typename VectorType>
struct perform_element_loop_functor {
	  typedef host_device_type                   device_type ;
	  typedef typename MatrixType::ScalarType Scalar;

	  perform_element_loop_functor(MatrixType* a_A, VectorType* a_b,
			  simple_mesh_description<GlobalOrdinal> a_mesh, h_v_global_ordinal a_elemIds,ElemData<GlobalOrdinal,Scalar> a_elem_data) :
				  A(a_A),b(a_b),mesh(a_mesh),elemIDs(a_elemIds)
	  //	  	  	  ,_elem_data(a_elem_data)
	  {};
	  MatrixType*  A ;
	  VectorType*  b ;
	  simple_mesh_description<GlobalOrdinal> mesh;
	  h_v_global_ordinal elemIDs;
     // ElemData<GlobalOrdinal,Scalar> _elem_data;
	  //--------------------------------------------------------------------------

	  inline
	  void operator()( const int i ) const
	  {
	        ElemData<GlobalOrdinal,Scalar> elem_data;// = _elem_data;

			compute_gradient_values(elem_data.grad_vals);
		    //Given an element-id, populate elem_data with the
		    //element's node_ids and nodal-coords:
		    get_elem_nodes_and_coords(mesh, elemIDs[i], elem_data);

		    //Next compute element-diffusion-matrix and element-source-vector:

		    compute_element_matrix_and_vector(elem_data);

		    //Now assemble the (dense) element-matrix and element-vector into the
		    //global sparse linear system:

		    sum_into_global_linear_system(elem_data, *A, *b);

	  };
};

template<typename GlobalOrdinal,
         typename MatrixType, typename VectorType>
void
perform_element_loop(const simple_mesh_description<GlobalOrdinal>& mesh,
                     const Box& local_elem_box,
                     MatrixType& A, VectorType& b,
                     Parameters& /*params*/)
{
  typedef typename MatrixType::ScalarType Scalar;

  int global_elems_x = mesh.global_box[0][1];
  int global_elems_y = mesh.global_box[1][1];
  int global_elems_z = mesh.global_box[2][1];

  //We will iterate the local-element-box (local portion of the mesh), and
  //get element-IDs in preparation for later assembling the FE operators
  //into the global sparse linear-system.

  GlobalOrdinal num_elems = get_num_ids<GlobalOrdinal>(local_elem_box);

  v_global_ordinal elemIDs("PerfElemLoop::elemIDs",num_elems);
  h_v_global_ordinal h_elemIDs = Kokkos::create_mirror_view(elemIDs);
  BoxIterator iter = BoxIterator::begin(local_elem_box);
  BoxIterator end  = BoxIterator::end(local_elem_box);


  for(size_t i=0; iter != end; ++iter, ++i) {
    h_elemIDs[i] = get_id<GlobalOrdinal>(global_elems_x, global_elems_y, global_elems_z,
                                       iter.x, iter.y, iter.z);
  }

  //Now do the actual finite-element assembly loop:

  ElemData<GlobalOrdinal,Scalar> elem_data;
  compute_gradient_values(elem_data.grad_vals);

  struct perform_element_loop_functor<GlobalOrdinal, MatrixType,VectorType> f(&A,&b,mesh,h_elemIDs,elem_data);
  Kokkos::parallel_for("perform_element_loop<Host>",h_elemIDs.dimension_0(),f);
  device_device_type::fence();
}

}//namespace miniFE

#endif

