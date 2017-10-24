
#ifndef _H_MINIFE_CUDA_UTILS
#define _H_MINIFE_CUDA_UTILS

#include <assert.h>
#include <shfl.h>
#include <device_atomic_functions.h>

__device__ __inline__ double miniFEAtomicAdd(double* address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val+__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

namespace miniFE {

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


#if defined(__CUDA_ARCH__) & (__CUDA_ARCH__ < 350)
template <class T> static __device__ inline T __ldg(T* ptr) { return *ptr; }
#endif

template <typename ValueType> 
  struct convert {
    union {
      ValueType v;
      int i;
    };
  };
template<typename ValueType> 
__device__ __inline__
ValueType __compare_and_swap_xor(ValueType val, int mask, int ASCENDING=true) {
  int laneId=threadIdx.x%32; //is there a better way to get this?
  int src=mask^laneId;

  convert<ValueType> new_val;
  new_val.v=val;
  new_val.i=__shfl(new_val.i,src);

  return (ASCENDING ^ (laneId<src) ^ (val<new_val.v)) ? val : new_val.v;
}


template<typename ValueType> 
__device__ __inline__
ValueType __sort(ValueType val, int ASCENDING=true) {
  int laneId=threadIdx.x%32;
  int DIRECTION=ASCENDING^(laneId/2%2);
  val=__compare_and_swap_xor(val,1,DIRECTION);
  DIRECTION=ASCENDING^(laneId/4%2);
  val=__compare_and_swap_xor(val,2,DIRECTION);
  val=__compare_and_swap_xor(val,1,DIRECTION);
  DIRECTION=ASCENDING^(laneId/8%2);
  val=__compare_and_swap_xor(val,4,DIRECTION);
  val=__compare_and_swap_xor(val,2,DIRECTION);
  val=__compare_and_swap_xor(val,1,DIRECTION);
  DIRECTION=ASCENDING^(laneId/16%2);
  val=__compare_and_swap_xor(val,8,DIRECTION);
  val=__compare_and_swap_xor(val,4,DIRECTION);
  val=__compare_and_swap_xor(val,2,DIRECTION);
  val=__compare_and_swap_xor(val,1,DIRECTION);
  DIRECTION=ASCENDING;
  val=__compare_and_swap_xor(val,16,DIRECTION);
  val=__compare_and_swap_xor(val,8,DIRECTION);
  val=__compare_and_swap_xor(val,4,DIRECTION);
  val=__compare_and_swap_xor(val,2,DIRECTION);
  val=__compare_and_swap_xor(val,1,DIRECTION);

  return val;
}

template<typename GlobalOrdinal>
  __device__ __inline__
  GlobalOrdinal lowerBound(const GlobalOrdinal *ptr, GlobalOrdinal low, GlobalOrdinal high, const GlobalOrdinal val) {

  //printf("Binary Search  for %d\n",val);
  
  while(high>=low)
  {
    GlobalOrdinal mid=low+(high-low)/2;
    GlobalOrdinal mval;
    mval=__ldg(ptr+mid);
    //printf("low: %d, high: %d, mid: %d, val: %d\n",low,high,mid, mval);
    if(mval>val)
      high=mid-1;
    else if (mval<val)
      low=mid+1;
    else
    {
      //printf("Found %d at index: %d\n", val, mid);
      return mid;
    }
  }
    
  if(__ldg(ptr+high) < val ) {
    //printf(" not found returning %d, (%d,%d)\n",high,low,high);
    return high;
  }
  else {
    //printf(" not found returning %d, (%d,%d)\n",high-1,low,high);
    return high-1;
  }
}

template<typename GlobalOrdinal>
  __device__ __inline__
  GlobalOrdinal binarySearch(const GlobalOrdinal *ptr, GlobalOrdinal low_, GlobalOrdinal high_, const GlobalOrdinal val) {

    GlobalOrdinal low=low_;
    GlobalOrdinal high=high_;

    //printf("%d:%d, Binary Search  for %d, low: %d, high: %d\n",threadIdx.x,val,val,low,high);

    while(high>=low)
    {
      GlobalOrdinal mid=low+(high-low)/2;
      GlobalOrdinal mval;
      mval=ptr[mid];
      //TODO: use ldg
      //mval=__ldg(ptr+mid);
      //printf("%d:%d, low: %d, high: %d, mid: %d, val: %d\n",threadIdx.x,val,low,high,mid, mval);
      if(mval>val)
        high=mid-1;
      else if (mval<val)
        low=mid+1;
      else
      {
       // printf("%d:%d, Found %d at index: %d\n", threadIdx.x,val, val, mid);
        return mid;
      }
    }
    //printf("%d,%d, not found\n",threadIdx.x,val);
    //not found
    return -1;  
}

class CudaManager {
  public:
    static void initialize() {
      if(!initialized) {
        cudaStreamCreate(&s1);
        cudaStreamCreate(&s2);
        cudaEventCreateWithFlags(&e1,cudaEventDisableTiming);
        cudaEventCreateWithFlags(&e2,cudaEventDisableTiming);
        initialized=true;
      }
    };
    static void finalize() {
      if(initialized) {
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
        initialized=false;
      }
    };
    static cudaStream_t s1;
    static cudaStream_t s2;
    static cudaEvent_t e1;
    static cudaEvent_t e2;
  private:
    static bool initialized;

};

template<class T> 
__global__ void cudaMemset_kernel(T* mem, T val, int N) {
  for(int idx=blockIdx.x*blockDim.x+threadIdx.x; idx<N; idx+=blockDim.x*gridDim.x) {
    mem[idx]=val;
  }
}

template<class T> 
__inline__ void cudaMemset_custom(T* mem, const T val, int N, cudaStream_t s) {
 int BLOCK_SIZE=512;
 int NUM_BLOCKS=min(8192,(N+BLOCK_SIZE-1)/BLOCK_SIZE);
  cudaMemset_kernel<<<NUM_BLOCKS,BLOCK_SIZE,0,s>>>(mem,val,N);
}

template<int Mark> 
__global__ void Marker_kernel() {}

template<int Mark>
void Marker() {
  Marker_kernel<Mark><<<1,1>>>();
}

}

#endif
