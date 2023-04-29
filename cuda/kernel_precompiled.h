// Access pre-compiled PTX kernel code.

#ifndef _KERNEL_PRECOMPILED_H_
#define _KERNEL_PRECOMPILED_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "defs.h"
#include "diamond.h"
#include "kernel.h"

namespace kernel_precompiled {

using defs::UV;

// Catch cuda errors when executing `expr`.
#define CURESULT(expr) CheckCUresult((expr), __FILE__, __LINE__, "Cuda error");
inline void CheckCUresult(CUresult res, const char *file, int line,
                          std::string msg) {
  if (res != CUDA_SUCCESS) {
    const char *errorname, *errorstring;
    cuGetErrorName(res, &errorname);
    cuGetErrorString(res, &errorstring);
    std::cerr << errorname << " (" << errorstring << ") in " << file << ":"
              << line << "\n";
    exit(res);
  }
}

struct PreCompiledKernelType {
  PreCompiledKernelType(bool is16bit, std::string capability, int npml,
                        bool withglobal = true, bool withshared = true,
                        bool withupdate = true)
      : is16bit(is16bit), capability(capability), npml(npml),
        withglobal(withglobal), withshared(withshared), withupdate(withupdate) {
  }

  std::string FileName(std::string dirname) {
    return dirname + "/kernel_" + Bits() + "_" + capability + "_" +
           std::to_string(npml) + "_" + Bool(withglobal) + Bool(withshared) +
           Bool(withupdate) + ".ptx";
  }

  std::string FunctionName() {
    if (is16bit)
      return "_ZN6kernel16SimulationKernelI7__half2fLi" + Npml() + Withs() +
             "EEEvNS_10KernelArgsIT_T0_EE";
    else
      return "_ZN6kernel16SimulationKernelIffLi" + Npml() + Withs() +
             "EEEvNS_10KernelArgsIT_T0_EE";
  }

  std::string Bits() { return (is16bit ? "16" : "32"); }
  std::string Npml() { return std::to_string(npml); }
  std::string Bool(bool val) { return (val ? "1" : "0"); }
  std::string Withs() {
    return "ELb" + Bool(withglobal) + "ELb" + Bool(withshared) + "ELb" +
           Bool(withupdate);
  }

  bool is16bit;
  std::string capability;
  int npml;
  bool withglobal, withshared, withupdate;
};

template <typename T>
PreCompiledKernelType
MakePreCompiledKernelType(std::string capability, int npml,
                          bool withglobal = true, bool withshared = true,
                          bool withupdate = true);

template <>
PreCompiledKernelType
MakePreCompiledKernelType<float>(std::string capability, int npml,
                                 bool withglobal, bool withshared,
                                 bool withupdate) {
  return PreCompiledKernelType(/*is16bit=*/false, capability, npml, withglobal,
                               withshared, withupdate);
}

template <>
PreCompiledKernelType
MakePreCompiledKernelType<half2>(std::string capability, int npml,
                                 bool withglobal, bool withshared,
                                 bool withupdate) {
  return PreCompiledKernelType(/*is16bit=*/true, capability, npml, withglobal,
                               withshared, withupdate);
}

#define CU_ERR(expr)                                                           \
  CudaAssertSuccess((expr), __FILE__, __LINE__, "Cuda error");
inline void CudaAssertSuccess(cudaError_t code, const char *file, int line,
                              std::string msg) {
  if (code != cudaSuccess) {
    std::cerr << msg << " (" << cudaGetErrorString(code) << ") in " << file
              << ":" << line << "\n";
    exit(code);
  }
}

// TODO: Consider relocating.
std::ostream &operator<<(std::ostream &os, const PreCompiledKernelType type) {
  os << "(" << type.is16bit << ", " << type.capability << ", " << type.npml
     << ", " << type.withglobal << ", " << type.withshared << ", "
     << type.withupdate << ")";
  return os;
}

void LaunchCooperativePreCompiledKernel(PreCompiledKernelType type, void *args,
                                        UV blockshape, UV gridshape,
                                        size_t sharedbytes,
                                        cudaStream_t stream = CU_STREAM_LEGACY,
                                        std::string dirname = "../ptx") {
  CUmodule mod;
  CUfunction fun;
  CURESULT(cuModuleLoad(&mod, type.FileName(dirname).c_str()));
  CURESULT(cuModuleGetFunction(&fun, mod, type.FunctionName().c_str()));
  const dim3 block(32, blockshape.u, blockshape.v),
      grid(1, gridshape.u, gridshape.v);
  CURESULT(cuLaunchCooperativeKernel(fun, grid.x, grid.y, grid.z, block.x,
                                     block.y, block.z, sharedbytes, stream,
                                     &args));
}

template <typename T, typename T1>
void RunKernel(PreCompiledKernelType type, kernel::KernelArgs<T, T1> args,
               cudaStream_t stream = CU_STREAM_LEGACY,
               std::string dirname = "../ptx") {
  int timesteps = defs::NumTimeSteps(args.rs.out);
  LaunchCooperativePreCompiledKernel(
      type, (void *)&args, args.rs.block, args.rs.grid,
      sizeof(T) * kernel::SharedElems(args.rs), stream, dirname);
}

} // namespace kernel_precompiled

#endif // _KERNEL_PRECOMPILED_H_
