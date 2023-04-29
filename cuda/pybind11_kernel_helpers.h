// This header extends kernel_helpers.h with the pybind11 specific interface to
// serializing descriptors. It also adds a pybind11 function for wrapping our
// custom calls in a Python capsule.
//
// Borrowed from github.com/dfm/extending-jax.
//

#ifndef _PYBIND11_KERNEL_HELPERS_H_
#define _PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

#include "kernel_helpers.h"

namespace pybind11_kernel_helpers {

// template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
//   return pybind11::bytes(kernel_helpers::PackDescriptorAsString(descriptor));
// }

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(kernel_helpers::bit_cast<void *>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

} // namespace pybind11_kernel_helpers

#endif
