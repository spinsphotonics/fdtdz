#ifndef _BUFFER_INDEXING_H_
#define _BUFFER_INDEXING_H_

#include "defs.h"
#include "diamond.h"

namespace buffer {
namespace {

using defs::kWarpSize;
using defs::RunShape;
using defs::UV;
using diamond::C;
using diamond::E;
using diamond::Ehc;
using diamond::H;
using diamond::N;
using diamond::Node;
using diamond::Nz;

// Number of either E- or H-elements (without taking into account "extra"
// points) for the section of a u- or v-edge that belongs to a warp.
static constexpr const int kWarpElems =
    (N / 2) * diamond::kNumXyz * Nz * kWarpSize;

// Number of elements at the tip of an edge (i.e. `Ey` and `(Hx, Hz)`).
__dhce__ int TipElems(Ehc ehc) { return (ehc == H ? 2 : 1) * Nz * kWarpSize; }

// Index used to move within an edge. The order of warps along a v-edge is
// reversed to hold to the convention that nodes are ordered in increasing
// y-index.
__dhce__ UV IntraEdgeIndex(UV warpindex, UV blockshape) {
  return UV(warpindex.v, blockshape.u - warpindex.u - 1);
}

struct SharedBlockIndexing {
  // For the shared buffer, tips are only stored in the v-buffers in general.
  __dhsc__ UV WarpStride(Ehc ehc) {
    return UV(kWarpElems - TipElems(ehc), kWarpElems);
  }

  // TODO: Check this documentation.
  // The ending tips are never included for E-field edges, but are always
  // included for H-field edges.
  __dhsc__ UV EdgeStride(UV blockshape, Ehc ehc) {
    return defs::VU(blockshape) * WarpStride(ehc) +
           (ehc == H ? UV(TipElems(ehc), 0) : UV(0, -TipElems(ehc)));
  }

  // Number of elements in the buffer.
  __dhsc__ UV NumElems(UV blockshape, Ehc ehc) {
    return (blockshape - 1) * EdgeStride(blockshape, ehc);
  }

  // Total number of elements in the buffer.
  __dhsc__ UV NumElems(UV blockshape) {
    return NumElems(blockshape, E) + NumElems(blockshape, H);
  }

  // Index where the shared buffer nodes for `warpindex` are located.
  __dhsc__ UV Index(UV warpindex, UV blockshape, Ehc ehc) {
    return
        // Place u-edge nodes first.
        UV(0, NumElems(blockshape, E).u + NumElems(blockshape, H).u) +

        // Place H-field nodes last.
        (ehc == H ? NumElems(blockshape, E) : UV(0, 0)) +

        // Initial offset which removes the starting tip for all edges
        // except
        // for the H-field v-edge, which is actually the ending tip of
        // the edge (and therefore, must be retained) as a consequence
        // of reversing the v-index as in `IntraEdgeIndex()`.
        -TipElems(ehc) * UV(1, ehc != H) +

        // Navigate to the correct edge.
        EdgeStride(blockshape, ehc) * warpindex +

        // Navigate to the correct warp within the edge.
        WarpStride(ehc) * IntraEdgeIndex(warpindex, blockshape);
  }
};

struct GlobalBlockIndexing {

  // For the global buffer, both edges contain tips.
  __dhsc__ UV WarpStride() { return UV(kWarpElems, kWarpElems); }

  // Only H-field v-edges need to have a beginning tip.
  __dhsc__ UV EdgeStride(UV blockshape, Ehc ehc) {
    return defs::VU(blockshape) * WarpStride() +
           (ehc == H ? UV(0, TipElems(ehc)) : UV(0, 0));
  }

  // Number of elements needed for a single block.
  __dhsc__ UV NumElems(UV blockshape, Ehc ehc) {
    return EdgeStride(blockshape, ehc);
  }

  // Total number of elements.
  __dhsc__ UV NumElems(UV blockshape) {
    return NumElems(blockshape, E) + NumElems(blockshape, H);
  }

  // Index where the global buffer nodes for `warpindex` are located.
  //
  // Note that this is much simpler than in the shared buffer case, because
  // there is only one u- and one v-edge associated with each block.
  //
  // Also note that we do not "merge" the u- and v-edge indexing. These must be
  // kept separate and independent at the block-level, being merged only when
  // the global buffer shape is known.
  //
  __dhsc__ UV Index(UV warpindex, UV blockshape, Ehc ehc) {
    return
        // Place H-field nodes last.
        (ehc == H ? NumElems(blockshape, E) : UV(0, 0)) +

        // Only H-field v-edges have a starting tip.
        -TipElems(ehc) * UV(1, ehc != H) +

        // Navigate to the correct warp within the edge.
        WarpStride() * IntraEdgeIndex(warpindex, blockshape);
  }
};

} // namespace
} // namespace buffer

#endif // _BUFFER_INDEXING_H_
