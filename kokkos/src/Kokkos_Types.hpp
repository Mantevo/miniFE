#include <KokkosCore_config.h>
#ifdef KOKKOS_HAVE_PTHREAD
  #include <Kokkos_Threads.hpp>
  typedef Kokkos::Threads host_device_type;
  #ifndef KOKKOS_HAVE_CUDA
    typedef Kokkos::Threads device_device_type;
  #endif
#else
  #ifdef KOKKOS_HAVE_OPENMP
    #include <Kokkos_OpenMP.hpp>
    typedef Kokkos::OpenMP host_device_type;
    #ifndef KOKKOS_HAVE_CUDA
      typedef Kokkos::OpenMP device_device_type;
    #endif
  #else
    #ifdef KOKKOS_HAVE_SERIAL
      #include <Kokkos_Serial.hpp>
      typedef Kokkos::Serial host_device_type;
      #ifndef KOKKOS_HAVE_CUDA
        typedef Kokkos::Serial device_device_type;
      #endif
    #else
      #error "No Kokkos Host Device defined"
    #endif
  #endif
#endif
#ifdef KOKKOS_HAVE_CUDA
  #include <Kokkos_Cuda.hpp>
  typedef Kokkos::Cuda device_device_type;
#endif

#include <Kokkos_View.hpp>

typedef int GlobalOrdinal;
typedef Kokkos::View<GlobalOrdinal*,device_device_type> v_global_ordinal;
typedef Kokkos::View<GlobalOrdinal*,device_device_type>::HostMirror h_v_global_ordinal;

