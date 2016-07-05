#ifndef _Hex8_box_utils_hpp_
#define _Hex8_box_utils_hpp_

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

#include <stdexcept>

#include <box_utils.hpp>
#include <ElemData.hpp>
#include <simple_mesh_description.hpp>
#include <Hex8.hpp>

namespace miniFE {


template<typename GlobalOrdinal>
__device__ __inline__
void get_hex8_node_ids(int nx, int ny,
                       GlobalOrdinal node0,
                       GlobalOrdinal* elem_node_ids)
{
//Given box dimensions nx and ny, and a starting node
//(local-node-0 for a hex8), compute the other nodes
//of the hex8 using the exodus ordering convention.
  elem_node_ids[0] = node0;
  elem_node_ids[1] = node0 + 1;
  elem_node_ids[2] = node0 + nx + 1;
  elem_node_ids[3] = node0 + nx;
  elem_node_ids[4] = node0 +     nx*ny;
  elem_node_ids[5] = node0 + 1 + nx*ny;
  elem_node_ids[6] = node0 + nx + nx*ny + 1;
  elem_node_ids[7] = node0 + nx + nx*ny;
}

template<typename Scalar>
__device__ __inline__
void get_hex8_node_coords_3d(Scalar x, Scalar y, Scalar z,
                             Scalar hx, Scalar hy, Scalar hz,
                             Scalar* elem_node_coords)
{
  //Input: x,y,z are the coordinates of local-node 0 for a Hex8.
  //'hx', 'hy', 'hz' are the lengths of the sides of the element
  //in each direction.

  elem_node_coords[0] = x;
  elem_node_coords[1] = y;
  elem_node_coords[2] = z;

  elem_node_coords[3] = x + hx;
  elem_node_coords[4] = y;
  elem_node_coords[5] = z;

  elem_node_coords[6] = x + hx;
  elem_node_coords[7] = y + hy;
  elem_node_coords[8] = z;

  elem_node_coords[9]  = x;
  elem_node_coords[10] = y + hy;
  elem_node_coords[11] = z;

  elem_node_coords[12] = x;
  elem_node_coords[13] = y;
  elem_node_coords[14] = z + hz;

  elem_node_coords[15] = x + hx;
  elem_node_coords[16] = y;
  elem_node_coords[17] = z + hz;

  elem_node_coords[18] = x + hx;
  elem_node_coords[19] = y + hy;
  elem_node_coords[20] = z + hz;

  elem_node_coords[21] = x;
  elem_node_coords[22] = y + hy;
  elem_node_coords[23] = z + hz;
}

template<typename MeshType, typename GlobalOrdinal, typename Scalar>
__inline__ __device__ 
void
get_elem_coords(const MeshType &mesh, GlobalOrdinal elemID, Scalar* node_coords)
{
  int global_nodes_x = mesh.global_elems_x+1;
  int global_nodes_y = mesh.global_elems_y+1;
  int global_nodes_z = mesh.global_elems_z+1;
  int elem_int_x, elem_int_y, elem_int_z;
  
  get_int_coords(elemID, mesh.global_elems_x, mesh.global_elems_y,mesh.global_elems_z,
             elem_int_x, elem_int_y, elem_int_z);
  
  GlobalOrdinal nodeID = get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, elem_int_x, elem_int_y, elem_int_z);
  
  Scalar ix,iy,iz;
  get_coords<GlobalOrdinal,Scalar>(nodeID, global_nodes_x, global_nodes_y, global_nodes_z,
                            ix,iy,iz);

  Scalar hx = 1.0/mesh.global_elems_x;
  Scalar hy = 1.0/mesh.global_elems_y;
  Scalar hz = 1.0/mesh.global_elems_z;

  get_hex8_node_coords_3d(ix, iy, iz, hx, hy, hz, node_coords);
}

template<typename MeshType, typename GlobalOrdinal, typename Scalar>
__inline__ __device__ 
void
get_elem_node_ids(const MeshType &mesh, GlobalOrdinal elemID, Scalar* node_ids)
{
  int global_nodes_x = mesh.global_elems_x+1;
  int global_nodes_y = mesh.global_elems_y+1;
  int global_nodes_z = mesh.global_elems_z+1;
  int elem_int_x, elem_int_y, elem_int_z;
  
  get_int_coords(elemID, mesh.global_elems_x, mesh.global_elems_y,mesh.global_elems_z,
             elem_int_x, elem_int_y, elem_int_z);
  
  GlobalOrdinal nodeID = get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, elem_int_x, elem_int_y, elem_int_z);
  get_hex8_node_ids(global_nodes_x, global_nodes_y, nodeID, node_ids);
      
  //Map node-IDs to rows because each processor may have a non-contiguous block of
  //node-ids, but needs a contiguous block of row-numbers:
  for(int i=0; i<Hex8::numNodesPerElem; ++i) {
    node_ids[i] = mesh.map_ids_to_rows.find_row_for_id(node_ids[i]);
  }
}


}//namespace miniFE

#endif

