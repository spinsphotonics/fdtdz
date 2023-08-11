// Infrastructure needed to serialize descriptors for the "opaque" parameter of
// the GPU custom call.
//
// Borrowed from github.com/dfm/extending-jax.
//

#ifndef _KERNEL_HELPERS_H_
#define _KERNEL_HELPERS_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace kernel_helpers {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

} // namespace kernel_helpers

#endif
