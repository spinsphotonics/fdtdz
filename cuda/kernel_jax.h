// Defines the FDTD simulation kernel.

#ifndef _KERNEL_JAX_H_
#define _KERNEL_JAX_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string.h>
#include <string>

#include "shapedefs.h"

namespace kernel_jax {

namespace {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bitcast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

} // namespace

using defs::RunShape;
using defs::UV;
using defs::XY;

struct KernelDescriptor {
  KernelDescriptor(std::string dirname, int capability, float dt, RunShape rs,
                   bool withglobal, bool withshared, bool withupdate)
      : dirname(dirname), capability(capability), dt(dt), rs(rs),
        withglobal(withglobal), withshared(withshared), withupdate(withupdate) {
  }

  std::string dirname;
  int capability;
  float dt;
  RunShape rs;
  bool withglobal, withshared, withupdate;

  static KernelDescriptor FromString(const char *str, std::size_t len) {
    KernelDescriptor kd = *bitcast<KernelDescriptor *>(str);
    kd.dirname = std::string(str + sizeof(KernelDescriptor),
                             len - sizeof(KernelDescriptor));
    return kd;
  }

  static std::string ToString(const KernelDescriptor &kd) {
    return std::string(bitcast<const char *>(&kd), sizeof(KernelDescriptor)) +
           kd.dirname;
  }
};

void kernel_f32(cudaStream_t stream, void **buffers, const char *opaque,
                std::size_t opaque_len);
void kernel_f16(cudaStream_t stream, void **buffers, const char *opaque,
                std::size_t opaque_len);

} // namespace kernel_jax

#endif // _KERNEL_JAX_H_
