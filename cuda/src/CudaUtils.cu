#include <CudaUtils.h>

namespace miniFE {

  cudaStream_t CudaManager::s1;
  cudaStream_t CudaManager::s2;
  cudaEvent_t CudaManager::e1;
  cudaEvent_t CudaManager::e2;
  bool CudaManager::initialized=false;

}
