// Helpers for unit tests.

#ifndef _TESTUTILS_H_
#define _TESTUTILS_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "buffer.h"
#include "cbuf.h"
#include "defs.h"
#include "diamond.h"
#include "field.h"
#include "kernel.h"
#include "kernel_jax.h"
#include "reference.h"
#include "slice.h"
#include "zcoeff.h"

// General testing utilities for making unit tests less painful.
namespace testutils {

using defs::RunShape;
using defs::UV;

// Catch cuda errors when executing `expr`.
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

// Super simple allocation that cleans up after itself. It's so stupid that it
// _cannot_ be passed as an argument (no deep copy). Uses managed memory so that
// both host and device code has access without having to do any memory
// transfers.
struct Alloc {
public:
  Alloc(int n) : ptr_(MakeAlloc(n)), n_(n) {}

  ~Alloc() { CU_ERR(cudaFree(ptr_)); }

  void *Ptr() const { return ptr_; }

  int NumBytes() const { return n_; }

private:
  void *ptr_;
  int n_;

  static void *MakeAlloc(int n) {
    void *ptr = NULL;
    CU_ERR(cudaMallocManaged(&ptr, n));
    return ptr;
  }
};

// Similar to `Alloc` but also allows for typed subscript operator. Useful for
// some unit tests where we want to be able to use `array[index]`.
template <typename T> struct Array {
public:
  Array(int n) : ptr_(MakeArray(n)), n_(n) {}

  ~Array() { CU_ERR(cudaFree(ptr_)); }

  T *Ptr() const { return ptr_; }

  T &operator[](int index) { return Ptr()[index]; }
  const T &operator[](int index) const { return Ptr()[index]; }

private:
  T *const ptr_;
  int n_;

  static T *MakeArray(int n) {
    void *ptr = NULL;
    CU_ERR(cudaMallocManaged(&ptr, sizeof(T) * n));
    return (T *)ptr;
  }
};

// Uses `cudaEvent` to time cuda operations (e.g. kernel execution).
struct Timer {
  Timer() {
    CU_ERR(cudaEventCreate(&start));
    CU_ERR(cudaEventCreate(&stop));
    CU_ERR(cudaEventRecord(start, NULL));
  }

  float end() {
    CU_ERR(cudaEventRecord(stop, NULL));
    CU_ERR(cudaEventSynchronize(stop));
    CU_ERR(cudaPeekAtLastError());
    CU_ERR(cudaDeviceSynchronize());
    float msecs_elapsed = 0.0f;
    CU_ERR(cudaEventElapsedTime(&msecs_elapsed, start, stop));
    return msecs_elapsed / 1e3;
  }

private:
  cudaEvent_t start, stop;
};

// Launches a cooperative kernel and returns the execution time in seconds.
float LaunchCooperativeKernel(void *kernel, void *args, UV blockshape,
                              UV gridshape, size_t sharedbytes = 0,
                              cudaStream_t stream = CU_STREAM_LEGACY) {
  // Shared memory per thread block.
  CU_ERR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedbytes));

  // Executes and times the kernel.
  dim3 blockdim(defs::kWarpSize, blockshape.u, blockshape.v);
  dim3 griddim(1, gridshape.u, gridshape.v);
  Timer timer;
  CU_ERR(cudaLaunchCooperativeKernel(kernel, griddim, blockdim, &args,
                                     sharedbytes, stream));
  CU_ERR(cudaPeekAtLastError());
  CU_ERR(cudaDeviceSynchronize());

  return timer.end();
}

// The number of registers needed for `kernel`.
size_t NumRegisters(void *kernel) {
  cudaFuncAttributes attrs;
  CU_ERR(cudaFuncGetAttributes(&attrs, kernel));
  return attrs.numRegs;
}

// The number of multiprocessors on device `deviceid`.
int NumMultiProcessors(int deviceid = 0) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceid);
  return prop.multiProcessorCount;
}

// Maximum allowable shared memory bytes per thread block.
int MaxSharedMemPerBlock(int deviceid = 0) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceid);
  return prop.sharedMemPerBlockOptin;
}

} // namespace testutils

namespace defs {

__dhce__ bool operator==(UV a, UV b) { return a.u == b.u && a.v == b.v; }

__dhce__ bool operator!=(UV a, UV b) { return !(a == b); }

__dhce__ bool operator==(XY a, XY b) { return a.x == b.x && a.y == b.y; }

__dhce__ bool operator==(XYT a, XYT b) {
  return a.x == b.x && a.y == b.y && a.t == b.t;
}

bool operator==(const RunShape::Pml &a, const RunShape::Pml &b) {
  return a.n == b.n && a.zshift == b.zshift;
}

bool operator==(const RunShape::Src &a, const RunShape::Src &b) {
  return a.type == b.type && a.pos == b.pos;
}

bool operator==(const RunShape::Out &a, const RunShape::Out &b) {
  return a.start == b.start && a.interval == b.interval && a.num == b.num;
}

bool operator==(const RunShape &a, const RunShape &b) {
  return a.block == b.block && a.grid == b.grid && a.spacing == b.spacing &&
         a.domain == b.domain && a.pml == b.pml && a.src == b.src &&
         a.out == b.out;
}

std::ostream &operator<<(std::ostream &os, const UV a) {
  os << "(" << a.u << ", " << a.v << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const XY a) {
  os << "(" << a.x << ", " << a.y << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const XYT a) {
  os << "(" << a.x << ", " << a.y << ", " << a.t << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const RunShape::Pml pml) {
  os << "(" << pml.n << ", " << pml.zshift << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const RunShape::Src::Type srctype) {
  if (srctype == RunShape::Src::YSLICE)
    os << "y";
  else // srctype == RunShape::Src::ZSLICE.
    os << "z";
  return os;
}

std::ostream &operator<<(std::ostream &os, const RunShape::Src src) {
  os << "(" << src.type << ", " << src.pos << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const RunShape::Out out) {
  os << "(" << out.start << ", " << out.interval << ", " << out.num << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const RunShape rs) {
  os << "(" << rs.block << ", " << rs.grid << ", " << rs.spacing << ", "
     << rs.domain << ", " << rs.pml << ", " << rs.src << ", " << rs.out << ")";
  return os;
}

} // namespace defs

namespace diamond {

struct NodeIterator {

  // Easily iterate over the nodes in a diamond. Can only be used in unit tests
  // (code that is either CPU-based or else does not need to be performant on
  // the GPU) because the `nvcc` compiler cannot convert this into performant
  // code.
  struct Iterator {
    __dhce__ Iterator(int nodeindex) : nodeindex(nodeindex) {}
    __dhce__ Iterator operator++() {
      ++nodeindex;
      return *this;
    }

    __dhce__ bool operator!=(const Iterator &other) const {
      return nodeindex != other.nodeindex;
    }

    __dhsc__ Ehc EhcFromIndex(int index) {
      if (index == 0)
        return E;
      else if (index == 1)
        return H;
      else // index == 2.
        return C;
    }

    __dhsc__ Xyz XyzFromIndex(int index) {
      if (index == 0)
        return X;
      else if (index == 1)
        return Y;
      else // index == 2.
        return Z;
    }

    __dhce__ const Node operator*() const {
      return Node(
          /*i=*/(nodeindex % (N + 2)) - 2,
          /*j=*/((nodeindex / (N + 2)) % (N + 1)) - 2,
          /*k=*/(nodeindex / ((N + 2) * (N + 1))) % Nz,
          /*ehc=*/
          EhcFromIndex((nodeindex / ((N + 2) * (N + 1) * Nz)) % kNumEhc),
          /*xyz=*/
          XyzFromIndex((nodeindex / ((N + 2) * (N + 1) * Nz * kNumEhc)) %
                       kNumXyz));
    }

  private:
    int nodeindex;
  };

  __dhsc__ Iterator begin() { return Iterator(0); }
  __dhsc__ Iterator end() {
    return Iterator((N + 2) * (N + 1) * Nz * kNumEhc * kNumXyz);
  }
};

__device__ constexpr const NodeIterator AllNodes;

// Useful for test functions.
__device__ __host__ bool operator==(Node a, Node b) {
  return a.i == b.i && a.j == b.j && a.k == b.k && a.ehc == b.ehc &&
         a.xyz == b.xyz;
}
__device__ __host__ bool operator!=(Node a, Node b) { return !(a == b); }

// Enables ordering for `std::set<Node>`.
bool operator<(Node a, Node b) {
  return a.i < b.i ||           //
         (a.i == b.i &&         //
          (a.j < b.j ||         //
           (a.j == b.j &&       //
            (a.k < b.k ||       //
             (a.k == b.k &&     //
              (a.ehc < b.ehc || //
               (a.ehc == b.ehc && a.xyz < b.xyz)))))));
}

std::ostream &operator<<(std::ostream &os, const Ehc ehc) {
  if (ehc == E)
    os << "E";
  else if (ehc == H)
    os << "H";
  else // ehc == C.
    os << "C";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Xyz xyz) {
  if (xyz == X)
    os << "x";
  else if (xyz == Y)
    os << "y";
  else // xyz == Z.
    os << "z";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Node n) {
  os << "(" << n.i << ", " << n.j << ", " << n.k << ", " << n.ehc << n.xyz
     << ")";
  return os;
}

} // namespace diamond

namespace reference {

template <typename T> void Fill(T val, T *ptr, int n) {
  for (int i = 0; i < n; ++i)
    ptr[i] = val;
}

// Allocate memory for a reference simulation.
template <typename T> struct SimAlloc {
public:
  SimAlloc(int x, int y, int z, int t, T dt, Node srcnode)
      : abs_(FieldElems(x, y)),    //
        mat_(FieldElems(x, y, z)), //
        zcoeff_(z),                //
        wf0_(t + 1),               //
        wf1_(t + 1),               //
        simparams_(x, y, z, abs_.Ptr(), mat_.Ptr(), zcoeff_.Ptr(), wf0_.Ptr(),
                   wf1_.Ptr(), dt, srcnode) {

    // Some standard values.
    Fill(Zero<T>(), simparams_.abs, FieldElems(x, y));
    Fill(One<T>(), simparams_.mat, FieldElems(x, y, z));
    Fill(ZCoeff<T>{One<T>(), Zero<T>(), Zero<T>(), //
                   One<T>(), Zero<T>(), Zero<T>()},
         simparams_.zcoeff, z);
  }

  SimParams<T> Params() { return simparams_; }

private:
  testutils::Array<T> abs_, mat_, wf0_, wf1_;
  testutils::Array<ZCoeff<T>> zcoeff_;
  SimParams<T> simparams_;
};

template <typename T>
void PrintZSlice(int k, Ehc ehc, Xyz xyz, int step, NodeType type,
                 SimParams<T> sp) {
  Cache<T> cache;
  for (int j = sp.y - 1; j >= 0; --j) {
    for (int i = 0; i < sp.x; ++i)
      std::cout << Get(Node(i, j, k, ehc, xyz), step, type, sp, cache) << " ";
    std::cout << "\n";
  }
}

} // namespace reference

namespace kernel {

using defs::RunShape;

// Allocates memory needed to run the kernel on GPU.
template <typename T, typename T1> struct KernelAlloc {
  // TODO: Change to have only one output for simplicity.
  static constexpr const int Nout = 2;

public:
  KernelAlloc(RunShape rs, T1 dt)                          //
      : intbuffer_(buffer::GlobalElems(rs)),               //
        intcbuffer_(cbuf::GlobalElems(rs.domain)),         //
        intmask_(slice::ZMask<T>::GlobalElems(rs.domain)), //
        intsrc_(rs.src.type == RunShape::Src::ZSLICE
                    ? slice::ZSrc<T>::GlobalElems(rs.domain)
                    : slice::YSrc<T>::GlobalElems(rs.domain.x)),
        extcbuffer_(cbuf::ExternalElems(rs.sub)),                //
        extabslayer_(slice::ZMask<T>::ExternalElems(rs.domain)), //
        extsrclayer_(
            rs.src.type == RunShape::Src::ZSLICE
                ? slice::ZSrc<T>::ExternalElems(rs.domain)
                : slice::YSrc<T>::ExternalElems(rs.domain.x, rs.pml.n)), //
        extwaveform_(2 * defs::NumTimeSteps(rs.out)),                    //
        zcoeff_(zcoeff::ExternalElems<T>(rs.pml.n)),                     //
        out_(field::ExternalElems<T>(rs.sub, rs.out.num, rs.pml.n)),
        args_(rs,
              KernelInternal(intbuffer_.Ptr(), intcbuffer_.Ptr(),
                             intmask_.Ptr(), intsrc_.Ptr()),
              KernelInputs(dt, extcbuffer_.Ptr(), extabslayer_.Ptr(),
                           extsrclayer_.Ptr(), extwaveform_.Ptr(),
                           zcoeff_.Ptr()),
              out_.Ptr()) {}

  KernelArgs<T, T1> Args() { return args_; }

private:
  testutils::Array<T> intbuffer_, intcbuffer_, intmask_, intsrc_;
  testutils::Array<T1> extcbuffer_, extabslayer_, extsrclayer_, extwaveform_,
      zcoeff_, out_;
  const KernelArgs<T, T1> args_;
};

// Launch a kernel.
template <typename T, typename T1>
__host__ float RunKernel(void *kernel, KernelArgs<T, T1> args, RunShape rs) {
  return testutils::LaunchCooperativeKernel(
      kernel, (void *)&args, rs.block, rs.grid, sizeof(T) * SharedElems(rs));
}

// Benchmark a kernel.
template <typename T, typename T1>
void BenchmarkKernel(void *kernel, kernel::KernelArgs<T, T1> args, RunShape rs,
                     int repeats) {
  // Manually collect the fastest runtime for our custom counters (TCUPS and
  // us/step).
  float minseconds = std::numeric_limits<float>::max();

  for (int iter = 0; iter < repeats; ++iter) {
    float seconds = kernel::RunKernel(kernel, args, rs);
    minseconds = seconds < minseconds ? seconds : minseconds;
  }

  // Use floating-point to avoid overflow.
  int timesteps = defs::NumTimeSteps(rs.out);
  float numsteps = float(scanner::NumSteps(timesteps, rs));
  float tcups =
      numsteps *
      float(defs::Prod(rs.block * rs.grid) *
            (diamond::N * diamond::N * defs::kWarpSize * diamond::Nz)) /
      minseconds / 1e12;
  float actual_tcups =
      float(timesteps) * float(rs.domain.x) * float(rs.domain.y) *
      float((defs::kWarpSize - rs.pml.n) * diamond::EffNz<T>()) / minseconds /
      1e12;

  // Print out results;
  std::cout << tcups << "/" << actual_tcups << " (raw/adj) TCUPS at " //
            << minseconds / numsteps * 1e6 << " us/step ("            //
            << minseconds * 1e3 << " ms, "                            //
            << numsteps << " steps, "                                 //
            << testutils::NumRegisters(kernel) << " regs)\n";
}

} // namespace kernel

namespace kernel_jax {

bool operator==(const KernelDescriptor &a, const KernelDescriptor &b) {
  return a.dirname == b.dirname && a.dt == b.dt && a.rs == b.rs &&
         a.withglobal == b.withglobal && a.withshared == b.withshared &&
         a.withupdate == b.withupdate;
}

std::ostream &operator<<(std::ostream &os, const KernelDescriptor kd) {
  os << "(" << kd.dirname << ", " << kd.dt << ", " << kd.rs << ", "
     << kd.withglobal << ", " << kd.withshared << ", " << kd.withupdate << ")";
  return os;
}

} // namespace kernel_jax

#endif // _TESTUTILS_H_
