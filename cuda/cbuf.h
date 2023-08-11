// Functions for using the C-node buffer.

#ifndef _CBUF_H_
#define _CBUF_H_

#include "buffer_ops.h"
#include "defs.h"
#include "diamond.h"
#include "slice.h"

namespace cbuf {

using buffer::Load;
using buffer::Node2;
using buffer::Node4;
using defs::kWarpSize;
using defs::RunShape;
using defs::UV;
using defs::XY;
using diamond::C;
using diamond::Cell;
using diamond::ExtZIndex;
using diamond::ExtZz;
using diamond::N;
using diamond::Node;
using diamond::Nz;
using diamond::X;
using diamond::Y;
using diamond::Z;

// External buffer does not hold c-values for auxiliary threads.
__dhce__ int ExternalElems(RunShape::Vol sub) {
  return (sub.x1 - sub.x0) * (sub.y1 - sub.y0) * (sub.z1 - sub.z0) *
         diamond::kNumXyz;
}

__dhce__ int ExternalIndex(Node n, RunShape::Vol sub) {
  int i = n.i - sub.x0;
  int j = n.j - sub.y0;
  int k = n.k - sub.z0;
  int xx = sub.x1 - sub.x0;
  int yy = sub.y1 - sub.y0;
  int zz = sub.z1 - sub.z0;
  return k + zz * (j + yy * (i + xx * diamond::Index(n.xyz)));
}

__dhce__ int GlobalElems(XY domain) {
  return domain.x * domain.y * kWarpSize * Nz * diamond::kNumXyz;
}

__dhce__ int GlobalIndex(Node n, XY domain) {
  if (n.xyz == Y) {
    return n.k + (kWarpSize * Nz) * (n.i + domain.x * n.j);
  } else {                                        // n.xyz == X || n.xyz == Z.
    return domain.x * domain.y * kWarpSize * Nz + //
           (n.k % Nz) +
           Nz * ((n.xyz == X ? 0 : 1) +
                 2 * ((n.k / Nz) +
                      kWarpSize *
                          (((n.xyz == Z) ? (n.i + domain.x - 1) % domain.x
                                         : n.i) +
                           domain.x * n.j)));
  }
}

__dhce__ int ClipToRange(int val, int lo, int hi) {
  if (val < lo)
    return lo;
  else if (val >= hi)
    return hi - 1;
  else
    return val;
}

__dhce__ Node NearestNode(Node n, RunShape::Vol sub) {
  return Node(ClipToRange(n.i, sub.x0, sub.x1), //
              ClipToRange(n.j, sub.y0, sub.y1), //
              ClipToRange(n.k, sub.z0, sub.z1), //
              n.ehc, n.xyz);
}

__dhce__ bool IsInside(Node n, RunShape::Vol v) {
  return n.i >= v.x0 && n.i < v.x1 && //
         n.j >= v.y0 && n.j < v.y1 && //
         n.k >= v.z0 && n.k < v.z1;
}

template <typename T>
__dh__ T GlobalValue(T *ptr, Node externalnode, bool isaux, RunShape::Vol sub,
                     RunShape::Vol vol) {
  if (isaux) {
    // Default value of `1` is needed to avoid branching in the update code when
    // dealing with auxiliary thread.
    return defs::One<T>();
  } else if (!IsInside(externalnode, vol)) { // Outside of volume.
    return defs::Zero<T>();
  } else if (IsInside(externalnode, sub)) { // Inside subvolume.
    return ptr[ExternalIndex(externalnode, sub)];
  } else { // Need to infer.
    Node nearestnode = NearestNode(externalnode, sub);
    return ptr[ExternalIndex(nearestnode, sub)];
  }
}

// Write the external node `n` into the internal buffer.
template <typename T>
__dh__ void WriteGlobal(T *src, T *dst, Node n, XY domain, int threadpos,
                        int npml, int zshift, bool isaux, RunShape::Vol sub,
                        RunShape::Vol vol, T *abs, T dt) {
  Node externalnode = n.K(ExtZIndex<T>(n.k, threadpos, npml, zshift));
  Node globalnode = n.dK(Nz * threadpos);
  // Apply the absorption mask.
  T absvalue = abs[slice::ZMask<T>::ExternalIndex(XY(n.i, n.j), n.xyz, domain)];
  T denom = (1 / dt) + (absvalue / 2);
  dst[GlobalIndex(globalnode, domain)] =
      GlobalValue(src, externalnode, isaux, sub, vol) / denom;
}

#ifndef __OMIT_HALF2__
// Fill the internal global c-buffer with values from the external buffer.
__dh__ void WriteGlobal(float *src, half2 *dst, Node n, XY domain,
                        int threadpos, int npml, int zshift, bool isaux,
                        RunShape::Vol sub, RunShape::Vol vol, float *abs,
                        float dt) {
  Node enodelo = n.K(ExtZIndex<half2>(n.k, threadpos, npml, zshift));
  Node enodehi = n.K(ExtZIndex<half2>(n.k + Nz, threadpos, npml, zshift));
  Node globalnode = n.dK(Nz * threadpos);
  // Apply the absorption mask.
  float absvalue =
      abs[slice::ZMask<float>::ExternalIndex(XY(n.i, n.j), n.xyz, domain)];
  float denom = (1 / dt) + (absvalue / 2);
  dst[GlobalIndex(globalnode, domain)] =
      __floats2half2_rn(GlobalValue(src, enodelo, isaux, sub, vol) / denom,
                        GlobalValue(src, enodehi, isaux, sub, vol) / denom);
}
#endif

// Two `Node`s. Useful for chunked memory access.
__dhce__ Node2 C2(int i, int j) {
  return Node2(Node(i, j, 0, C, Y), Node(i, j, 1, C, Y));
}

// Four `Node`s.
__dhce__ Node4 C4(int i, int j) {
  return Node4(Node(i, j, 0, C, X), Node(i, j, 1, C, X),
               Node(i + 1, j, 0, C, Z), Node(i + 1, j, 1, C, Z));
}

template <typename T>
__dhce__ void LoadGlobal(Cell<T> &cell, uint2 *ptr2, XY offset, XY domain) {
  Load(cell, C2(offset.x, offset.y),
       ptr2[kWarpSize * (offset.x + domain.x * offset.y)]);
}

template <typename T>
__dhce__ void LoadGlobal(Cell<T> &cell, uint4 *ptr4, XY offset0, XY offset1,
                         XY domain) {
  Load(cell, C4(offset0.x, offset0.y),
       ptr4[kWarpSize * (offset0.x + domain.x * offset0.y)]);
  Load(cell, C4(offset1.x, offset1.y),
       ptr4[kWarpSize * (offset1.x + domain.x * offset1.y)]);
}

template <typename T>
__dhce__ void LoadGlobal(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                         XY pos, XY domain) {
  if (diamond::IsDiamondCompletelyInDomain(pos, domain)) {
    uint2 *ptr2 =
        ((uint2 *)(ptr +
                   GlobalIndex(Node(pos.x, pos.y, /*k=*/0, C, Y), domain))) +
        threadpos;
    if (warppos.u == 0 || warppos.v == 0) // Leading point.
      LoadGlobal(cell, ptr2, /*offset=*/XY(1, 0), domain);
    if (warppos.u == 0) // U-edge.
      LoadGlobal(cell, ptr2, /*offset=*/XY(0, 1), domain);
    if (warppos.v == 0) // V-edge.
      LoadGlobal(cell, ptr2, /*offset=*/XY(0, -1), domain);

    uint4 *ptr4 =
        ((uint4 *)(ptr +
                   GlobalIndex(Node(pos.x, pos.y, /*k=*/0, C, X), domain))) +
        threadpos;
    if (warppos.u == 0) // U-edge.
      LoadGlobal(cell, ptr4, XY(0, 1), XY(-1, 2), domain);
    if (warppos.v == 0) // V-edge
      LoadGlobal(cell, ptr4, XY(-1, -1), XY(0, 0), domain);
  }
}

struct SharedIndexing {
  // Only store tips in the v-buffers.
  static constexpr const UV WarpStride =
      Nz * kWarpSize * ((N / 2) * diamond::kNumXyz - UV(1, 0));

  // Index used to move within an edge. The order of warps along a v-edge is
  // reversed to hold to the convention that nodes are ordered in increasing
  // y-index.
  __dhsc__ UV IntraEdgeIndex(UV warpindex, UV blockshape) {
    return UV(warpindex.v, blockshape.u - warpindex.u - 1);
  }

  // Do not store outermost tip of v-buffers.
  __dhsc__ UV EdgeStride(UV blockshape) {
    return defs::VU(blockshape) * WarpStride - UV(0, Nz * kWarpSize);
  }

  // Number of elements in the buffer.
  __dhsc__ UV NumElems(UV blockshape) {
    return (blockshape - 1) * EdgeStride(blockshape);
  }

  // Index where the shared buffer nodes for `warpindex` are located.
  __dhsc__ UV Index(UV warpindex, UV blockshape) {
    return
        // Place u-edge nodes first.
        UV(0, NumElems(blockshape).u) +

        // Navigate to the correct edge.
        EdgeStride(blockshape) * warpindex +

        // Navigate to the correct warp within the edge.
        WarpStride * IntraEdgeIndex(warpindex, blockshape);
  }
};

__dhce__ int SharedElems(UV blockshape) {
  return defs::Sum(SharedIndexing::NumElems(blockshape));
}

template <typename T>
__dhce__ void LoadShared(Cell<T> &cell, T *ptr, int threadpos, //
                         Node4 n40, Node2 n20, Node4 n41, Node2 n21,
                         bool includelast) {
  Load(cell, n40, ((uint4 *)ptr)[threadpos]);
  Load(cell, n20, ((uint2 *)ptr)[2 * kWarpSize + threadpos]);
  Load(cell, n41, ((uint4 *)ptr)[3 * (kWarpSize / 2) + threadpos]);
  if (includelast)
    Load(cell, n21, ((uint2 *)ptr)[5 * kWarpSize + threadpos]);
}

template <typename T>
__dhce__ void LoadShared(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                         UV blockshape) {
  if (warppos.u > 0) // U-edge.
    LoadShared(
        cell, ptr + SharedIndexing::Index(warppos - UV(1, 0), blockshape).u,
        threadpos, C4(0, 1), C2(0, 1), C4(-1, 2), /*never used*/ C2(0, 1),
        /*includelast=*/false);

  if (warppos.v > 0) // V-edge.
    LoadShared(cell,
               ptr + SharedIndexing::Index(warppos - UV(0, 1), blockshape).v,
               threadpos, C4(-1, -1), C2(0, -1), C4(0, 0), C2(1, 0),
               /*includelast=*/warppos.u > 0);
}

template <typename T>
__dhce__ void StoreShared(Cell<T> &cell, T *ptr, int threadpos, //
                          Node2 n20, Node4 n40, Node2 n21, Node4 n41,
                          bool includefirst) {
  if (includefirst)
    Store(cell, n20, ((uint2 *)ptr) - kWarpSize + threadpos);
  Store(cell, n40, ((uint4 *)ptr) + threadpos);
  Store(cell, n21, ((uint2 *)ptr) + 2 * kWarpSize + threadpos);
  Store(cell, n41, ((uint4 *)ptr) + 3 * (kWarpSize / 2) + threadpos);
}

template <typename T>
__dhce__ void StoreShared(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                          UV blockshape) {
  if (warppos.u < blockshape.u - 1) // U-edge.
    StoreShared(cell, ptr + SharedIndexing::Index(warppos, blockshape).u,
                threadpos,
                /*never used*/ C2(-1, -1), C4(-1, -1), C2(-1, -1), C4(-2, 0),
                /*includefirst=*/false);

  if (warppos.v < blockshape.v - 1) // V-edge.
    StoreShared(cell, ptr + SharedIndexing::Index(warppos, blockshape).v,
                threadpos, C2(-2, 0), C4(-2, 1), C2(-1, 1), C4(-1, 2),
                /*includefirst=*/warppos.u < blockshape.u - 1);
}

template <typename T, typename T1>
__dh__ void Convert(T1 *src, T *dst, RunShape rs, int zshift, int threadpos,
                    UV warppos, UV blockpos, T1 *abs, T1 dt) {
  // NOTE: We abuse the `UV` notation to iterate in (x, y) coordinates.
  UV init = warppos + rs.block * blockpos;
  UV stride = rs.block * rs.grid;
  for (int i = init.u; i < rs.domain.x; i += stride.u)
    for (int j = init.v; j < rs.domain.y; j += stride.v)
      for (int k = 0; k < diamond::Nz; ++k)
        for (diamond::Xyz xyz : diamond::AllXyz)
          cbuf::WriteGlobal(src, dst, diamond::Node(i, j, k, diamond::E, xyz),
                            rs.domain, threadpos, rs.pml.n, zshift,
                            defs::IsAux(threadpos, rs.pml.n), rs.sub, rs.vol,
                            abs, dt);
}

} // namespace cbuf

#endif // _CBUF_H_
