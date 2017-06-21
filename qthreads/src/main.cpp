
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
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>

#include <miniFE_version.h>

#include <outstream.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <Box.hpp>
#include <BoxPartition.hpp>
#include <box_utils.hpp>
#include <Parameters.hpp>
#include <utils.hpp>
#include <driver.hpp>
#include <YAML_Doc.hpp>
#include <param_utils.hpp>

#include <qthread/qthread.h>

#if MINIFE_INFO != 0
#include <miniFE_info.hpp>
#else
#include <miniFE_no_info.hpp>
#endif

//The following macros should be specified as compile-macros in the
//makefile. They are defaulted here just in case...
#ifndef MINIFE_SCALAR
#define MINIFE_SCALAR double
#endif
#ifndef MINIFE_LOCAL_ORDINAL
#define MINIFE_LOCAL_ORDINAL int
#endif
#ifndef MINIFE_GLOBAL_ORDINAL
#define MINIFE_GLOBAL_ORDINAL int
#endif

// ************************************************************************

void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params);
void add_configuration_to_yaml(YAML_Doc& doc, int numprocs, int numthreads);
void add_timestring_to_yaml(YAML_Doc& doc);
void add_qthreads_configuration_to_yaml(YAML_Doc& doc, int argc, char * argv[]);

inline void print_box(int myproc, const char* name, const Box& box,
                      const char* name2, const Box& box2)
{
  std::cout << "proc " << myproc << " "<<name
      <<" ("<<box[0][0]<<","<<box[0][1]<<") "
      <<" ("<<box[1][0]<<","<<box[1][1]<<") "
      <<" ("<<box[2][0]<<","<<box[2][1]<<") "
      <<name2
      <<" ("<<box2[0][0]<<","<<box2[0][1]<<") "
      <<" ("<<box2[1][0]<<","<<box2[1][1]<<") "
      <<" ("<<box2[2][0]<<","<<box2[2][1]<<") "<<std::endl;
}

//
//We will create a 'box' of size nx X ny X nz, partition it among processors,
//then call miniFE::driver which will use the partitioned box as the domain
//from which to assemble finite-element matrices into a global matrix and
//vector, then solve the linear-system using Conjugate Gradients.
//

int main(int argc, char** argv) {
  miniFE::Parameters params;
  miniFE::get_parameters(argc, argv, params);

  int numprocs = 1, myproc = 0;
  miniFE::initialize_mpi(argc, argv, numprocs, myproc);

  miniFE::timer_type start_time = miniFE::mytimer();

#ifdef MINIFE_DEBUG
  outstream(numprocs, myproc);
#endif

  std::cout << "Initializing qthreads library...";
  int init_result = qthread_initialize();
  if(qthread_initialize() == 0) {
	std::cout << "successful." << std::endl;
  } else {
	std::cout << "error, library did not return successful." << std::endl;
	exit(-1);
  }

  //make sure each processor has the same parameters:
  miniFE::broadcast_parameters(params);


  Box global_box = { 0, params.nx, 0, params.ny, 0, params.nz };
  std::vector<Box> local_boxes(numprocs);

  box_partition(0, numprocs, 2, global_box, &local_boxes[0]);

  Box& my_box = local_boxes[myproc];

//print_box(myproc, "global-box", global_box, "local-box", my_box);

  std::ostringstream osstr;
  osstr << "miniFE." << params.nx << "x" << params.ny << "x" << params.nz;
#ifdef HAVE_MPI
  osstr << ".P"<<numprocs;
#endif
  osstr << ".";
  if (params.name != "") osstr << params.name << ".";

  YAML_Doc doc("miniFE", MINIFE_VERSION, ".", osstr.str());
  if (myproc == 0) {
    add_params_to_yaml(doc, params);
    add_configuration_to_yaml(doc, numprocs, params.numthreads);
    add_timestring_to_yaml(doc);
    add_qthreads_configuration_to_yaml(doc, argc, argv);
  }

  //Most of the program is performed in the 'driver' function, which is
  //templated on < Scalar, LocalOrdinal, GlobalOrdinal >.
  //To run miniFE with float instead of double, or 'long long' instead of int,
  //etc., change these template-parameters by changing the macro definitions in
  //the makefile or on the make command-line.

  miniFE::driver< MINIFE_SCALAR, MINIFE_LOCAL_ORDINAL, MINIFE_GLOBAL_ORDINAL>(global_box, my_box, params, doc);

  miniFE::timer_type total_time = miniFE::mytimer() - start_time;

  if (myproc == 0) {
    doc.add("Total Program Time",total_time);
    doc.generateYAML();
  }

  miniFE::finalize_mpi();

  // Clean up we're done.
  qthread_finalize();

  return 0;
}

void add_params_to_yaml(YAML_Doc& doc, miniFE::Parameters& params)
{
  doc.add("Global Run Parameters","");
  doc.get("Global Run Parameters")->add("dimensions","");
  doc.get("Global Run Parameters")->get("dimensions")->add("nx",params.nx);
  doc.get("Global Run Parameters")->get("dimensions")->add("ny",params.ny);
  doc.get("Global Run Parameters")->get("dimensions")->add("nz",params.nz);
  doc.get("Global Run Parameters")->add("load_imbalance", params.load_imbalance);
  if (params.mv_overlap_comm_comp == 1) {
    std::string val("1 (yes)");
    doc.get("Global Run Parameters")->add("mv_overlap_comm_comp", val);
  }
  else {
    std::string val("0 (no)");
    doc.get("Global Run Parameters")->add("mv_overlap_comm_comp", val);
  }
}

void add_configuration_to_yaml(YAML_Doc& doc, int numprocs, int numthreads)
{
  doc.get("Global Run Parameters")->add("number of processors", numprocs);

  doc.add("Platform","");
  doc.get("Platform")->add("hostname",MINIFE_HOSTNAME);
  doc.get("Platform")->add("kernel name",MINIFE_KERNEL_NAME);
  doc.get("Platform")->add("kernel release",MINIFE_KERNEL_RELEASE);
  doc.get("Platform")->add("processor",MINIFE_PROCESSOR);

  doc.add("Build","");
  doc.get("Build")->add("CXX",MINIFE_CXX);
#if MINIFE_INFO != 0
  doc.get("Build")->add("compiler version",MINIFE_CXX_VERSION);
#endif
  doc.get("Build")->add("CXXFLAGS",MINIFE_CXXFLAGS);
  std::string using_mpi("no");
#ifdef HAVE_MPI
  using_mpi = "yes";
#endif
  doc.get("Build")->add("using MPI",using_mpi);
}

void add_timestring_to_yaml(YAML_Doc& doc)
{
  std::time_t rawtime;
  struct tm * timeinfo;
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::ostringstream osstr;
  osstr.fill('0');
  osstr << timeinfo->tm_year+1900 << "-";
  osstr.width(2); osstr << timeinfo->tm_mon+1 << "-";
  osstr.width(2); osstr << timeinfo->tm_mday << ", ";
  osstr.width(2); osstr << timeinfo->tm_hour << "-";
  osstr.width(2); osstr << timeinfo->tm_min << "-";
  osstr.width(2); osstr << timeinfo->tm_sec;
  std::string timestring = osstr.str();
  doc.add("Run Date/Time",timestring);
}

void add_qthreads_configuration_to_yaml(YAML_Doc& doc, int argc, char *argv[])
{
  int num_shepherds = qthread_readstate(ACTIVE_SHEPHERDS);
  int num_workers = qthread_readstate(ACTIVE_WORKERS);
  int stack_size = qthread_readstate(STACK_SIZE);

  std::string arg_string;
  Mantevo::read_args_into_string(argc, argv, arg_string);
  std::string scheduler =
    Mantevo::parse_parameter<std::string>(arg_string, "scheduler",
                                          "none-specified");
  std::string topology =
    Mantevo::parse_parameter<std::string>(arg_string, "topology",
                                          "none-specified");

  doc.add("Qthreads","");
  doc.get("Qthreads")->add("num shepherds", num_shepherds);
  doc.get("Qthreads")->add("num workers", num_workers);
  doc.get("Qthreads")->add("num wps", num_workers/num_shepherds);
  doc.get("Qthreads")->add("stack size", stack_size);
  doc.get("Qthreads")->add("scheduler", scheduler);
  doc.get("Qthreads")->add("topology", topology);
}

