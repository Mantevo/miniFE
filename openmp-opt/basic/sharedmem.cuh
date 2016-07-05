/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a 
// simple wrapper struct that will declare the extern array with a different 
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
// 
// To use dynamically allocated shared memory in a templatized __global__ or 
// __device__ function, just replace code like this:
//
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata) 
//  {
//      // Shared mem size is determined by the host app at run time
//      extern __shared__  T sdata[];
//      ...
//      doStuff(sdata);
//      ...
//   }
//  
//   With this
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata) 
//  {
//      // Shared mem size is determined by the host app at run time
//      SharedMemory<T> smem;
//      T* sdata = smem.getPointer();
//      ...
//      doStuff(sdata);
//      ...
//   }
//****************************************************************************

// This is the un-specialized struct.  Note that we prevent instantiation of this 
// struct by putting an undefined symbol in the function body so it won't compile.
template <typename T>
struct SharedMemory
{
    // Ensure that we won't compile any un-specialized types
    __device__ T* getPointer() {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedMemory <int>
{
    __device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }    
};

template <>
struct SharedMemory <unsigned int>
{
    __device__ unsigned int* getPointer() { extern __shared__ unsigned int s_uint[]; return s_uint; }    
};

template <>
struct SharedMemory <char>
{
    __device__ char* getPointer() { extern __shared__ char s_char[]; return s_char; }    
};

template <>
struct SharedMemory <unsigned char>
{
    __device__ unsigned char* getPointer() { extern __shared__ unsigned char s_uchar[]; return s_uchar; }    
};

template <>
struct SharedMemory <short>
{
    __device__ short* getPointer() { extern __shared__ short s_short[]; return s_short; }    
};

template <>
struct SharedMemory <unsigned short>
{
    __device__ unsigned short* getPointer() { extern __shared__ unsigned short s_ushort[]; return s_ushort; }    
};

template <>
struct SharedMemory <long>
{
    __device__ long* getPointer() { extern __shared__ long s_long[]; return s_long; }    
};

template <>
struct SharedMemory <unsigned long>
{
    __device__ unsigned long* getPointer() { extern __shared__ unsigned long s_ulong[]; return s_ulong; }    
};

template <>
struct SharedMemory <bool>
{
    __device__ bool* getPointer() { extern __shared__ bool s_bool[]; return s_bool; }    
};

template <>
struct SharedMemory <float>
{
    __device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }    
};

template <>
struct SharedMemory <double>
{
    __device__ double* getPointer() { extern __shared__ double s_double[]; return s_double; }    
};


#endif //_SHAREDMEM_H_
