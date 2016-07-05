/*
//@HEADER
// ************************************************************************
// 
//   Kokkos: Manycore Performance-Portable Multidimensional Arrays
//              Copyright (2012) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov) 
// 
// ************************************************************************
//@HEADER
*/

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#include <Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_Cuda_Internal.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {


void cuda_internal_error_throw( cudaError e , const char * name, const char * file, const int line )
{
  std::ostringstream out ;
  out << name << " error: " << cudaGetErrorString(e);
  if (file) {
    out << " " << file << ":" << line;
  }
  throw_runtime_exception( out.str() );
}

//----------------------------------------------------------------------------
// Some significant cuda device properties:
//
// cudaDeviceProp::name                : Text label for device
// cudaDeviceProp::major               : Device major number
// cudaDeviceProp::minor               : Device minor number
// cudaDeviceProp::warpSize            : number of threads per warp
// cudaDeviceProp::multiProcessorCount : number of multiprocessors
// cudaDeviceProp::sharedMemPerBlock   : capacity of shared memory per block
// cudaDeviceProp::totalConstMem       : capacity of constant memory
// cudaDeviceProp::totalGlobalMem      : capacity of global memory
// cudaDeviceProp::maxGridSize[3]      : maximum grid size

//
//  Section 4.4.2.4 of the CUDA Toolkit Reference Manual
//
// struct cudaDeviceProp {
//   char name[256];
//   size_t totalGlobalMem;
//   size_t sharedMemPerBlock;
//   int regsPerBlock;
//   int warpSize;
//   size_t memPitch;
//   int maxThreadsPerBlock;
//   int maxThreadsDim[3];
//   int maxGridSize[3];
//   size_t totalConstMem;
//   int major;
//   int minor;
//   int clockRate;
//   size_t textureAlignment;
//   int deviceOverlap;
//   int multiProcessorCount;
//   int kernelExecTimeoutEnabled;
//   int integrated;
//   int canMapHostMemory;
//   int computeMode;
//   int concurrentKernels;
//   int ECCEnabled;
//   int pciBusID;
//   int pciDeviceID;
//   int tccDriver;
//   int asyncEngineCount;
//   int unifiedAddressing;
//   int memoryClockRate;
//   int memoryBusWidth;
//   int l2CacheSize;
//   int maxThreadsPerMultiProcessor;
// };


namespace {



class CudaInternalDevices {
public:
  enum { MAXIMUM_DEVICE_COUNT = 8 };
  struct cudaDeviceProp  m_cudaProp[ MAXIMUM_DEVICE_COUNT ] ;
  int                    m_cudaDevCount ;

  CudaInternalDevices();

  static const CudaInternalDevices & singleton();
};

CudaInternalDevices::CudaInternalDevices()
{
  // See 'cudaSetDeviceFlags' for host-device thread interaction
  // Section 4.4.2.6 of the CUDA Toolkit Reference Manual

  CUDA_SAFE_CALL (cudaGetDeviceCount( & m_cudaDevCount ) );

  for ( int i = 0 ; i < m_cudaDevCount ; ++i ) {
    CUDA_SAFE_CALL( cudaGetDeviceProperties( m_cudaProp + i , i ) );
  }
}

const CudaInternalDevices & CudaInternalDevices::singleton()
{
  static CudaInternalDevices self ; return self ;
}

}

//----------------------------------------------------------------------------

class CudaInternal {
private:

  CudaInternal( const CudaInternal & );
  CudaInternal & operator = ( const CudaInternal & );

public:

  typedef Cuda::size_type size_type ;

  int         m_cudaDev ;
  unsigned    m_maxWarpCount ;
  unsigned    m_maxBlock ;
  unsigned    m_maxSharedWords ;
  size_type   m_scratchSpaceCount ;
  size_type   m_scratchFlagsCount ;
  size_type   m_scratchUnifiedCount ;
  size_type   m_scratchUnifiedSupported ;
  size_type * m_scratchSpace ;
  size_type * m_scratchFlags ;
  size_type * m_scratchUnified ;

  static CudaInternal & raw_singleton();
  static CudaInternal & singleton();

  const CudaInternal & assert_initialized() const ;

  int is_initialized() const
    { return 0 != m_scratchSpace && 0 != m_scratchFlags ; }

  void initialize( int cuda_device_id );
  void finalize();

  void print_configuration( std::ostream & ) const ;

  ~CudaInternal();

  CudaInternal()
    : m_cudaDev( -1 )
    , m_maxWarpCount( 0 )
    , m_maxBlock( 0 ) 
    , m_maxSharedWords( 0 )
    , m_scratchSpaceCount( 0 )
    , m_scratchFlagsCount( 0 )
    , m_scratchUnifiedCount( 0 )
    , m_scratchUnifiedSupported( 0 )
    , m_scratchSpace( 0 )
    , m_scratchFlags( 0 )
    , m_scratchUnified( 0 )
    {}

  size_type * scratch_space( const size_type size );
  size_type * scratch_flags( const size_type size );
  size_type * scratch_unified( const size_type size );
};

//----------------------------------------------------------------------------


void CudaInternal::print_configuration( std::ostream & s ) const
{
  const CudaInternalDevices & dev_info = CudaInternalDevices::singleton();

#if defined( KOKKOS_HAVE_CUDA )
    s << "macro  KOKKOS_HAVE_CUDA      : defined" << std::endl ;
#endif
#if defined( KOKKOS_HAVE_CUDA_ARCH )
    s << "macro  KOKKOS_HAVE_CUDA_ARCH = " << KOKKOS_HAVE_CUDA_ARCH
      << " = capability " << KOKKOS_HAVE_CUDA_ARCH / 100
      << "." << ( KOKKOS_HAVE_CUDA_ARCH % 100 ) / 10
      << std::endl ;
#endif
#if defined( CUDA_VERSION )
    s << "macro  CUDA_VERSION          = " << CUDA_VERSION
      << " = version " << CUDA_VERSION / 1000
      << "." << ( CUDA_VERSION % 1000 ) / 10
      << std::endl ;
#endif

  for ( int i = 0 ; i < dev_info.m_cudaDevCount ; ++i ) {
    s << "Kokkos::Cuda[ " << i << " ] "
      << dev_info.m_cudaProp[i].name
      << " capability " << dev_info.m_cudaProp[i].major << "." << dev_info.m_cudaProp[i].minor
      << ", Total Global Memory: " << human_memory_size(dev_info.m_cudaProp[i].totalGlobalMem) 
      << ", Shared Memory per Block: " << human_memory_size(dev_info.m_cudaProp[i].sharedMemPerBlock);
    if ( m_cudaDev == i ) s << " : Selected" ;
    s << std::endl ;
  }
}

//----------------------------------------------------------------------------

CudaInternal::~CudaInternal()
{
  if ( m_scratchSpace ||
       m_scratchFlags ||
       m_scratchUnified ) {
    std::cerr << "Kokkos::Cuda ERROR: Failed to call Kokkos::Cuda::finalize()"
              << std::endl ;
    std::cerr.flush();
  }
}

CudaInternal & CudaInternal::raw_singleton()
{ static CudaInternal self ; return self ; }

const CudaInternal & CudaInternal::assert_initialized() const
{
  if ( m_cudaDev == -1 ) {
    const std::string msg("CATASTROPHIC FAILURE: Using Kokkos::Cuda before calling Kokkos::Cuda::initialize(...)");
    throw_runtime_exception( msg );
  }
  return *this ;
}

CudaInternal & CudaInternal::singleton()
{
  CudaInternal & s = raw_singleton();
  s.assert_initialized();
  return s ;
}

void CudaInternal::initialize( int cuda_device_id )
{
  enum { WordSize = sizeof(size_type) };

  if ( ! Cuda::host_mirror_device_type::is_initialized() ) {
    const std::string msg("Cuda::initialize ERROR : Cuda::host_mirror_device_type is not initialized");
    throw_runtime_exception( msg );
  }

  const CudaInternalDevices & dev_info = CudaInternalDevices::singleton();

  const bool ok_init = 0 == m_scratchSpace || 0 == m_scratchFlags ;

  const bool ok_id   = 0 <= cuda_device_id &&
                            cuda_device_id < dev_info.m_cudaDevCount ;

  // Need device capability 2.0 or better

  const bool ok_dev = ok_id &&
    ( 2 <= dev_info.m_cudaProp[ cuda_device_id ].major &&
      0 <= dev_info.m_cudaProp[ cuda_device_id ].minor );

  if ( ok_init && ok_dev ) {

    const struct cudaDeviceProp & cudaProp =
      dev_info.m_cudaProp[ cuda_device_id ];

    m_cudaDev = cuda_device_id ;

    CUDA_SAFE_CALL( cudaSetDevice( m_cudaDev ) );
    CUDA_SAFE_CALL( cudaDeviceReset() );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    //----------------------------------
    // Maximum number of warps,
    // at most one warp per thread in a warp for reduction.

    // HCE 2012-February :
    // Found bug in CUDA 4.1 that sometimes a kernel launch would fail
    // if the thread count == 1024 and a functor is passed to the kernel.
    // Copying the kernel to constant memory and then launching with
    // thread count == 1024 would work fine.
    //
    // HCE 2012-October :
    // All compute capabilities support at least 16 warps (512 threads).
    // However, we have found that 8 warps typically gives better performance.

    m_maxWarpCount = 8 ;

    // m_maxWarpCount = cudaProp.maxThreadsPerBlock / Impl::CudaTraits::WarpSize ;

    if ( Impl::CudaTraits::WarpSize < m_maxWarpCount ) {
      m_maxWarpCount = Impl::CudaTraits::WarpSize ;
    }

    m_maxSharedWords = cudaProp.sharedMemPerBlock / WordSize ;

    //----------------------------------

    m_maxBlock = cudaProp.maxGridSize[0] ;

    //----------------------------------

    m_scratchUnifiedSupported = cudaProp.unifiedAddressing ;

    if ( ! m_scratchUnifiedSupported ) {
      std::cout << "Kokkos::Cuda device "
                << cudaProp.name << " capability "
                << cudaProp.major << "." << cudaProp.minor
                << " does not support unified virtual address space"
                << std::endl ;
    }

    //----------------------------------
    // Multiblock reduction uses scratch flags for counters
    // and scratch space for partial reduction values.
    // Allocate some initial space.  This will grow as needed.

    {
      const unsigned reduce_block_count = m_maxWarpCount * Impl::CudaTraits::WarpSize ;

      (void) scratch_unified( 16 * sizeof(size_type) );
      (void) scratch_flags( reduce_block_count * 2  * sizeof(size_type) );
      (void) scratch_space( reduce_block_count * 16 * sizeof(size_type) );
    }
  }
  else {

    std::ostringstream msg ;
    msg << "Kokkos::Cuda::initialize(" << cuda_device_id << ") FAILED" ;

    if ( ! ok_init ) {
      msg << " : Already initialized" ;
    }
    if ( ! ok_id ) {
      msg << " : Device identifier out of range "
          << "[0.." << dev_info.m_cudaDevCount << "]" ;
    }
    else if ( ! ok_dev ) {
      msg << " : Device " ;
      msg << dev_info.m_cudaProp[ cuda_device_id ].major ;
      msg << "." ;
      msg << dev_info.m_cudaProp[ cuda_device_id ].minor ;
      msg << " has insufficient capability, required 2.0 or better" ;
    }
    Kokkos::Impl::throw_runtime_exception( msg.str() );
  } 
}

//----------------------------------------------------------------------------

typedef Cuda::size_type ScratchGrain[ Impl::CudaTraits::WarpSize ] ;
enum { sizeScratchGrain = sizeof(ScratchGrain) };


Cuda::size_type *
CudaInternal::scratch_flags( const Cuda::size_type size )
{
  assert_initialized();

  if ( m_scratchFlagsCount * sizeScratchGrain < size ) {

    Cuda::memory_space::decrement( m_scratchFlags );
  
    m_scratchFlagsCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

    m_scratchFlags = (size_type *)
      Cuda::memory_space::allocate(
        std::string("InternalScratchFlags") ,
        typeid( ScratchGrain ),
        sizeof( ScratchGrain ),
        m_scratchFlagsCount );

    CUDA_SAFE_CALL( cudaMemset( m_scratchFlags , 0 , m_scratchFlagsCount * sizeScratchGrain ) );
  }

  return m_scratchFlags ;
}

Cuda::size_type *
CudaInternal::scratch_space( const Cuda::size_type size )
{
  assert_initialized();

  if ( m_scratchSpaceCount * sizeScratchGrain < size ) {

    Cuda::memory_space::decrement( m_scratchSpace );
  
    m_scratchSpaceCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

    m_scratchSpace = (size_type *)
      Cuda::memory_space::allocate(
        std::string("InternalScratchSpace") ,
        typeid( ScratchGrain ),
        sizeof( ScratchGrain ),
        m_scratchSpaceCount );
  }

  return m_scratchSpace ;
}

Cuda::size_type *
CudaInternal::scratch_unified( const Cuda::size_type size )
{
  assert_initialized();

  if ( m_scratchUnifiedSupported ) {

    const bool allocate   = m_scratchUnifiedCount * sizeScratchGrain < size ;
    const bool deallocate = m_scratchUnified && ( 0 == size || allocate );

    if ( allocate || deallocate ) {
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    if ( deallocate ) {

      CUDA_SAFE_CALL( cudaFreeHost( m_scratchUnified ) );

      m_scratchUnified = 0 ;
      m_scratchUnifiedCount = 0 ;
    }

    if ( allocate ) {

      m_scratchUnifiedCount = ( size + sizeScratchGrain - 1 ) / sizeScratchGrain ;

      CUDA_SAFE_CALL( cudaHostAlloc( (void **)( & m_scratchUnified ) ,
                      m_scratchUnifiedCount * sizeScratchGrain ,
                      cudaHostAllocDefault ) );
    }
  }

  return m_scratchUnified ;
}

//----------------------------------------------------------------------------

void CudaInternal::finalize()
{
  if ( 0 != m_scratchSpace || 0 != m_scratchFlags ) {

    Cuda::memory_space::decrement( m_scratchSpace );
    Cuda::memory_space::decrement( m_scratchFlags );
    (void) scratch_unified( 0 );

    m_cudaDev            = -1 ;
    m_maxWarpCount       = 0 ;
    m_maxBlock           = 0 ; 
    m_maxSharedWords     = 0 ;
    m_scratchSpaceCount  = 0 ;
    m_scratchFlagsCount  = 0 ;
    m_scratchSpace       = 0 ;
    m_scratchFlags       = 0 ;
  }
}

//----------------------------------------------------------------------------

Cuda::size_type cuda_internal_maximum_warp_count()
{ return CudaInternal::singleton().m_maxWarpCount ; }

Cuda::size_type cuda_internal_maximum_grid_count()
{ return CudaInternal::singleton().m_maxBlock ; }

Cuda::size_type cuda_internal_maximum_shared_words()
{ return CudaInternal::singleton().m_maxSharedWords ; }

Cuda::size_type * cuda_internal_scratch_space( const Cuda::size_type size )
{ return CudaInternal::singleton().scratch_space( size ); }

Cuda::size_type * cuda_internal_scratch_flags( const Cuda::size_type size )
{ return CudaInternal::singleton().scratch_flags( size ); }

Cuda::size_type * cuda_internal_scratch_unified( const Cuda::size_type size )
{ return CudaInternal::singleton().scratch_unified( size ); }


} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

Cuda::size_type Cuda::detect_device_count()
{ return Impl::CudaInternalDevices::singleton().m_cudaDevCount ; }

int Cuda::is_initialized()
{ return Impl::CudaInternal::raw_singleton().is_initialized(); }

void Cuda::initialize( const Cuda::SelectDevice config )
{ Impl::CudaInternal::raw_singleton().initialize( config.cuda_device_id ); }

std::vector<unsigned>
Cuda::detect_device_arch()
{
  const Impl::CudaInternalDevices & s = Impl::CudaInternalDevices::singleton();

  std::vector<unsigned> output( s.m_cudaDevCount );

  for ( int i = 0 ; i < s.m_cudaDevCount ; ++i ) {
    output[i] = s.m_cudaProp[i].major * 100 + s.m_cudaProp[i].minor ;
  }

  return output ;
}

Cuda::size_type Cuda::device_arch()
{
  const int dev_id = Impl::CudaInternal::singleton().m_cudaDev ;

  const struct cudaDeviceProp & cudaProp =
    Impl::CudaInternalDevices::singleton().m_cudaProp[ dev_id ] ;

  return cudaProp.major * 100 + cudaProp.minor ;
}

void Cuda::finalize()
{ Impl::CudaInternal::raw_singleton().finalize(); }

void Cuda::print_configuration( std::ostream & s , const bool )
{ Impl::CudaInternal::raw_singleton().print_configuration( s ); }

bool Cuda::sleep() { return false ; }

bool Cuda::wake() { return true ; }

void Cuda::fence()
{ 
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

unsigned Cuda::team_max()
{
  return Impl::CudaInternal::singleton().m_maxWarpCount << Impl::CudaTraits::WarpIndexShift ;
}

} // namespace Kokkos

//----------------------------------------------------------------------------

