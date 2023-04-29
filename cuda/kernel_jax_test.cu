#include <gtest/gtest.h>

#include "kernel_jax.h"
#include "testutils.h"

namespace kernel_jax {
namespace {

using defs::RunShape;
using defs::UV;
using defs::XY;

TEST(KernelDescriptor, PackAndUnpack) {
  KernelDescriptor kd("path/to/dir", 75, 1.0f,
                      RunShape(UV(2, 2), UV(2, 2), 2, XY(2, 2)), true, true,
                      true);
  std::string str = KernelDescriptor::ToString(kd);
  EXPECT_EQ(KernelDescriptor::FromString(str.c_str(), str.length()), kd);
}

} // namespace
} // namespace kernel_jax
