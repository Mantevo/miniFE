#include<Kokkos_Core.hpp>
#include<Kokkos_CrsMatrix.hpp>
#include<Kokkos_MV.hpp>

typedef Kokkos::DefaultHostExecutionSpace host_device_type;
typedef Kokkos::DefaultExecutionSpace device_device_type;

typedef int GlobalOrdinal;
typedef Kokkos::View<GlobalOrdinal*,device_device_type> v_global_ordinal;
typedef Kokkos::View<GlobalOrdinal*,device_device_type>::HostMirror h_v_global_ordinal;

