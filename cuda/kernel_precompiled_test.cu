// Used to benchmark the kernel.

#include <gtest/gtest.h>

#include "cuda.h"
#include "kernel_precompiled.h"

namespace kernel_precompiled {
namespace {

TEST(KernelPreCompiled, FileName) {
  EXPECT_EQ(
      PreCompiledKernelType(/*is16bit=*/true, /*capability=*/"75", /*npml=*/6)
          .FileName("../ptx"),
      "../ptx/kernel_16_75_6_111.ptx");
  EXPECT_EQ(
      PreCompiledKernelType(/*is16bit=*/false, /*capability=*/"37", /*npml=*/2)
          .FileName("../ptx"),
      "../ptx/kernel_32_37_2_111.ptx");
}

TEST(KernelPreCompiled, FunctionName) {
  EXPECT_EQ(
      PreCompiledKernelType(/*is16bit=*/true, /*capability=*/"75", /*npml=*/6)
          .FunctionName(),
      "_ZN6kernel16SimulationKernelI7__half2fLi6ELb1ELb1ELb1EEEvNS_"
      "10KernelArgsIT_T0_EE");
  EXPECT_EQ(
      PreCompiledKernelType(/*is16bit=*/false, /*capability=*/"37", /*npml=*/2)
          .FunctionName(),
      "_ZN6kernel16SimulationKernelIffLi2ELb1ELb1ELb1EEEvNS_10KernelArgsIT_T0_"
      "EE");
}

} // namespace
} // namespace kernel_precompiled
