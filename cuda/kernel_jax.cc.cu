#include <cassert>
#include <string>

#include "kernel.h"
#include "kernel_helpers.h"
#include "kernel_jax.h"
#include "kernel_precompiled.h"
#include "scanner.h"
#include "testutils.h"

namespace kernel_jax {

using defs::RunShape;
using defs::UV;
using defs::XY;
using kernel_precompiled::PreCompiledKernelType;

namespace {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
__global__ void EmptyKernel(kernel::KernelArgs<T, float> args){};

template <typename T>
inline void ApplyKernel(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len) {
  const KernelDescriptor kd = KernelDescriptor::FromString(opaque, opaque_len);

  if (!scanner::IsValidRunShape(kd.rs))
    throw std::invalid_argument("Invalid run shape!");

  kernel::KernelInputs<float> kernelinputs(
      /*dt=*/kd.dt,
      /*cbuffer=*/reinterpret_cast<float *>(buffers[0]),
      /*abslayer=*/reinterpret_cast<float *>(buffers[1]),
      /*srclayer=*/reinterpret_cast<float *>(buffers[2]),
      /*waveform=*/reinterpret_cast<float *>(buffers[3]),
      /*zcoeff=*/reinterpret_cast<float *>(buffers[4]));

  kernel::KernelInternal<T> kernelinternal(
      /*buffer=*/reinterpret_cast<T *>(buffers[5]),
      /*cbuffer=*/reinterpret_cast<T *>(buffers[6]),
      /*mask=*/reinterpret_cast<T *>(buffers[7]),
      /*src=*/reinterpret_cast<T *>(buffers[8]));

  float *kerneloutput = reinterpret_cast<float *>(buffers[9]);

  kernel::KernelArgs<T, float> args(kd.rs, kernelinternal, kernelinputs,
                                    kerneloutput);

  kernel_precompiled::RunKernel(
      kernel_precompiled::MakePreCompiledKernelType<T>(
          std::to_string(kd.capability), kd.rs.pml.n, kd.withglobal,
          kd.withshared, kd.withupdate),
      args, stream, kd.dirname);

  ThrowIfError(cudaGetLastError());
}

} // namespace

void kernel_f32(cudaStream_t stream, void **buffers, const char *opaque,
                std::size_t opaque_len) {
  ApplyKernel<float>(stream, buffers, opaque, opaque_len);
}

void kernel_f16(cudaStream_t stream, void **buffers, const char *opaque,
                std::size_t opaque_len) {
  ApplyKernel<half2>(stream, buffers, opaque, opaque_len);
}

} // namespace kernel_jax
