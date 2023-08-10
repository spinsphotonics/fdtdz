// Used to benchmark the kernel.

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "buffer.h"
#include "defs.h"
#include "diamond.h"
#include "kernel.h"
#include "kernel_precompiled.h"
#include "scanner.h"
#include "testutils.h"

namespace kernel {
namespace {

using defs::RunShape;
using defs::UV;
using defs::XY;
using diamond::N;

// Benchmark a kernel.
template <typename T, typename T1>
void BenchmarkKernel(kernel_precompiled::PreCompiledKernelType type,
                     KernelArgs<T, T1> args, int repeats) {
  // Manually collect the fastest runtime for our custom counters (TCUPS and
  // us/step).
  float minseconds = std::numeric_limits<float>::max();

  for (int iter = 0; iter < repeats; ++iter) {
    testutils::Timer timer;
    kernel_precompiled::RunKernel(type, args);
    float seconds = timer.end();
    minseconds = seconds < minseconds ? seconds : minseconds;
  }

  // Use floating-point to avoid overflow.
  int timesteps = defs::NumTimeSteps(args.rs.out);
  float numsteps = float(scanner::NumSteps(timesteps, args.rs));
  float tcups =
      numsteps *
      float(defs::Prod(args.rs.block * args.rs.grid) *
            (diamond::N * diamond::N * defs::kWarpSize * diamond::Nz)) /
      minseconds / 1e12;
  float actual_tcups =
      float(timesteps) * float(args.rs.domain.x) * float(args.rs.domain.y) *
      float((defs::kWarpSize - args.rs.pml.n) * diamond::EffNz<T>()) /
      minseconds / 1e12;

  // Print out results;
  std::cout << tcups << "/" << actual_tcups << " (raw/adj) TCUPS at " //
            << minseconds / numsteps * 1e6 << " us/step ("            //
            << minseconds * 1e3 << " ms, "                            //
            << numsteps << " steps)\n";                               //
  // << testutils::NumRegisters(kernel) << " regs)\n";
}

void SpacingBenchmark(int timesteps, int repeats, std::vector<int> spacings,
                      bool withglobal, bool withshared, bool withupdate) {
  const int npml = 7;

  for (int spacing : spacings) {
    RunShape rs(
        /*blockshape=*/UV(2, 4),
        /*gridshape=*/UV(6, 6),
        /*spacing=*/spacing,
        /*domain=*/XY(200, 216),
        /*pml=*/RunShape::Pml(/*n=*/npml, /*zshift=*/2),
        /*src=*/RunShape::Src(RunShape::Src::YSLICE, /*srcpos=*/64),
        /*out=*/
        RunShape::Out(
            /*start=*/timesteps, /*interval=*/1, /*num=*/1),
        /*sub=*/
        RunShape::Vol(N, 200 - N, N, 216 - N, 0, diamond::ExtZz<half2>(npml)),
        /*vol=*/
        RunShape::Vol(N, 200 - N, N, 216 - N, 0, diamond::ExtZz<half2>(npml)));

    // /*x=*/RunShape::Out::Range(0, 200),
    // /*y=*/RunShape::Out::Range(0, 216),
    // /*z=*/RunShape::Out::Range(0, diamond::ExtZz<half2>(npml))));
    ASSERT_TRUE(scanner::IsValidRunShape(rs)) << "Invalid run shape: " << rs;

    KernelAlloc<half2, float> alloc(rs, /*dt=*/defs::One<float>());
    std::cout << spacing << ": ";
    BenchmarkKernel<half2>(
        kernel_precompiled::MakePreCompiledKernelType<half2>(
            /*capability=*/"75", npml, withglobal, withshared, withupdate),
        alloc.Args(), repeats);
  }
}

// Expects `TESTNAME`, `WITHGLOBAL`, `WITHSHARED`, and `WITHUPDATE` to be
// compile-time flags.
TEST(Kernel, Benchmark) {
  const int timesteps = 1000;
  const int repeats = 3;
  std::vector<int> spacings = {1, 2, 4, 8, 12};
  std::cout << "\nUpdate only\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/false,
                   /*withshared=*/false, /*withupdate=*/true);
  std::cout << "\nShared only\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/false,
                   /*withshared=*/true, /*withupdate=*/false);
  std::cout << "\nGlobal only\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/true,
                   /*withshared=*/false, /*withupdate=*/false);
  std::cout << "\nUpdate + Shared\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/false,
                   /*withshared=*/true, /*withupdate=*/true);
  std::cout << "\nUpdate + Global\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/true,
                   /*withshared=*/false, /*withupdate=*/true);
  std::cout << "\nShared + Global\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/true,
                   /*withshared=*/true, /*withupdate=*/false);
  std::cout << "\nUnified\n===\n";
  SpacingBenchmark(timesteps, repeats, spacings, /*withglobal=*/true,
                   /*withshared=*/true, /*withupdate=*/true);
  std::cout << "\n";
}

} // namespace
} // namespace kernel
