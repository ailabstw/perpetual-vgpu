/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
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

#define __USE_GNU

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstring>

// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface function
extern "C" { void* __libc_dlsym (void *map, const char *name); }

// We need to give the pre-processor a chance to replace a function, such as:
// cuMemAlloc => cuMemAlloc_v2
#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

// We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA driver symbols will bypass
// the hooking mechanism (this includes the CUDA runtime). Its tricky though, since if we replace the
// real dlsym with ours, we can't dlsym() the real dlsym. To get around that, call the 'private'
// libc interface called __libc_dlsym to get the real dlsym.
typedef void* (*fnDlsym)(void*, const char*);
static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

/*
 ** interposed functions
 */
void *dlsym(void *handle, const char *symbol) {
  // Early out if not a CUDA driver symbol
  if (strncmp(symbol, "cu", 2) != 0) {
    return (real_dlsym(handle, symbol));
  }

  if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
    return (void *)(&cuMemAlloc);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAllocManaged)) == 0) {
    return (void *)(&cuMemAllocManaged);
  }
  
  return (real_dlsym(handle, symbol));
}

// Proxy cuMemAlloc to cuMemAllocManaged
CUresult CUDAAPI cuMemAlloc (CUdeviceptr * dptr, size_t bytesize) {
  static void *cuMemAllocManaged_func = (void *)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(cuMemAllocManaged));
  return ((CUresult CUDAAPI(*) (CUdeviceptr* dptr, size_t bytesize, unsigned int flags))cuMemAllocManaged_func)(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
}
