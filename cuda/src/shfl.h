#if defined(__CUDA_ARCH__) & (__CUDA_ARCH__ < 300)

#define MAX_BLOCK 256
template <class T>
__device__ inline T __shfl_down(T var, const unsigned int delta, const unsigned int width=32) {
  __shared__ volatile T shfl_array[MAX_BLOCK];
  unsigned int x=threadIdx.x%width;
  unsigned int wg=x%32/width;

  shfl_array[threadIdx.y*blockDim.x+wg*width+x]=var;
  unsigned int srcLane=x+delta;
  if(srcLane<width) {
    var=shfl_array[threadIdx.y*blockDim.x+wg*width+srcLane];
  }

  return var;
}

template <class T>
__device__ inline T __shfl_up(T var, const unsigned int delta, const unsigned int width=32) {
  __shared__ volatile T shfl_array[MAX_BLOCK];
  unsigned int x=threadIdx.x%width;
  unsigned int wg=x%32/width;

  shfl_array[threadIdx.y*blockDim.x+wg*width+x]=var;
  unsigned int srcLane=x-delta;
  if(srcLane<width) {
    var=shfl_array[threadIdx.y*blockDim.x+wg*width+srcLane];
  }

  return var;
}

template <class T>
__device__ inline T __shfl(T var, const unsigned int srcLane, const unsigned int width=32) {
  __shared__ volatile T shfl_array[MAX_BLOCK];
  unsigned int x=threadIdx.x%width;
  unsigned int wg=x%32/width;

  shfl_array[threadIdx.y*blockDim.x+wg*width+x]=var;
  if(srcLane<width)
    var=shfl_array[threadIdx.y*blockDim.x+wg*width+srcLane];

  return var;
}

template <class T>
__device__ inline T __shfl_xor(T var, const unsigned int laneMask, const unsigned int width=32) {
  __shared__ volatile T shfl_array[MAX_BLOCK];
  unsigned int x=threadIdx.x%width;
  unsigned int wg=x%32/width;
  unsigned int srcLane=laneMask^x;

  shfl_array[threadIdx.y*blockDim.x+wg*width+x]=var;
  if(srcLane<width)
    var=shfl_array[threadIdx.y*blockDim.x+wg*width+srcLane];

  return var;
}
#endif
