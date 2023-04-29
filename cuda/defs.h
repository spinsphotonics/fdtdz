// Basic definitions.

#ifndef _DEFS_H_
#define _DEFS_H_

#include <cuda_fp16.h>

#include "macros.h"
#include "shapedefs.h"

namespace defs {

// Defines a `0` even when `T = half2`.
template <typename T> __dh__ T Zero() { return T{0}; }

// Defines a `1` even when `T = half2`.
template <typename T> __dh__ T One() { return T{1}; }

// Allows for a standard conversion between `float` and `half2`.
template <typename T, typename T1> __dhce__ T Convert(T1 value) {
  return T(value);
}

#ifndef __OMIT_HALF2__
template <> __dh__ half2 Zero<half2>() { return __floats2half2_rn(0.0f, 0.0f); }
template <> __dh__ half2 One<half2>() { return __floats2half2_rn(1.0f, 1.0f); }
template <> half2 Convert(float value) { return __float2half2_rn(value); }
#endif

// Assumes 32 threads/warp, used to map each warp to the simulation z-axis.
constexpr const int kWarpSize = 32;

// Maps the CUDA thread hierarchy to a hierarchical diamond-like structure.
//
// In this structure, each warp is a diamond-shaped pillar which then form
// diamonds at both the block and grid levels. Here we use the term "diamond"
// loosely, since, in practice, these will actually be rotated rectangles since
// the `u`- and `v`-dimensions of the diamond do not need to be equal.
//
// This mapping is implemented by mapping the thread and block index dimensions
// to the "diamond" dimensions as `(x, y, z) --> (z, u, v)`.
//
__device__ int ThreadPos() { return threadIdx.x; }
__device__ UV WarpPos() { return UV(threadIdx.y, threadIdx.z); }
__device__ UV BlockPos() { return UV(blockIdx.y, blockIdx.z); }
__device__ UV BlockShape() { return UV(blockDim.y, blockDim.z); }
__device__ UV GridShape() { return UV(gridDim.y, gridDim.z); }

// Determines if a warp has the lead/trail position in a block.
__dhce__ bool IsLeadU(UV warppos) { return warppos.u == 0; }
__dhce__ bool IsLeadV(UV warppos) { return warppos.v == 0; }
__dhce__ bool IsTrailU(UV warppos, UV blockshape) {
  return warppos.u == blockshape.u - 1;
}
__dhce__ bool IsTrailV(UV warppos, UV blockshape) {
  return warppos.v == blockshape.v - 1;
}

// Determines whether a thread serves as storage for auxiliary PML variables
// only.
__dhce__ bool IsAux(int tpos, int npml) { return tpos >= kWarpSize - npml; }
__dhce__ bool IsAux(int tpos, RunShape rs) { return IsAux(tpos, rs.pml.n); }

__dhce__ int NumTimeSteps(RunShape::Out out) {
  return out.start + (out.num - 1) * out.interval + 1;
}

}; // namespace defs

#endif // _DEFS_H_
