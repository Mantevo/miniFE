#ifndef miniFE_info_hpp
#define miniFE_info_hpp

#define MINIFE_HOSTNAME "white23"
#define MINIFE_KERNEL_NAME "'Linux'"
#define MINIFE_KERNEL_RELEASE "'3.10.0-327.el7.ppc64le'"
#define MINIFE_PROCESSOR "'ppc64le'"

#define MINIFE_CXX "'/home/projects/pwr8-rhel72/ibm/clang/20160914/clang/bin/clang++'"
#define MINIFE_CXX_VERSION "'clang version 4.0.0 (bbot:/home/bbot/repos/clang.git 36692876ee67dbbe648874e79144e523fdaf2ce5) (bbot:/home/bbot/repos/llvm.git 2ce3658b3be59a2be93850ddee9792d3265e18a1)'"
#define MINIFE_CXXFLAGS "'-v -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=/home/projects/pwr8-rhel72/cuda/8.0.27 -ffp-contract=fast -mcpu=power8 -mtune=power8 -fslp-vectorize-aggressive -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Xcuda-ptxas -maxrregcount=32 '"

#endif
