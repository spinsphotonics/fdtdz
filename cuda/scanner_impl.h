// Coordinates how each diamond traverses the simulation domain.

#ifndef _SCANNER_IMPL_H_
#define _SCANNER_IMPL_H_

#include "defs.h"
#include "diamond.h"

namespace scanner {

using defs::RunShape;
using defs::UV;
using defs::XY;
using defs::XYT;

// Number of cells along the u- and v-axes of the diamond.
__dhce__ UV Diamond(RunShape rs) {
  return (diamond::N / 2) * rs.block * rs.grid;
}

// Shape of the diamond with the blockspacing included.
//
// We ignore the fact that since the spacing is only along the x-axis, the
// resulting shape is not really a diamond.
//
__dhce__ UV SpacedDiamond(RunShape rs) {
  return Diamond(rs) + rs.grid * rs.spacing;
}

// Diamond position in the simulation domain at `step = 0`.
//
// Places the trailing diamond on the `(xx, yy)` point of the simulation domain
// to ensure that all values will be computable as we move forward.
//
__dhce__ XYT StartDomainPos(UV w, UV b, RunShape rs) {
  // Reverse the indices because we want to start from the trailing point.
  UV revb = (rs.grid - 1) - b;
  UV revw = (rs.block - 1) - w;

  UV d = revw + rs.block * revb;
  XYT offset = (diamond::N / 2) * XYT(d.u + d.v, d.u - d.v, 0);
  XYT spacingoffset = rs.spacing * (revb.u + revb.v) * XYT(1, 0, 1);
  return XYT(rs.domain.x, rs.domain.y, 0) + offset + spacingoffset;
}

// Wrap-around the x-boundary.
__dhce__ XYT WrapX(XYT p, RunShape rs) {
  int n = p.x / rs.domain.x;
  return p + n * XYT(-rs.domain.x, Diamond(rs).v, Diamond(rs).v - rs.domain.x);
}

// Wrap-around the y-boundary.
__dhce__ XYT WrapY(XYT p, RunShape rs) {
  int n = p.y / rs.domain.y;
  // TODO: Document the factor of `2` which is related to the "diagonal lines of
  // constant time".
  return p + n * XYT(0, -rs.domain.y, 2 * Diamond(rs).u - rs.domain.y);
}

// Returns a wrapped `XYT` where `(x, y)` is in the simulation domain.
//
// Assumes that `p.x >= 0` and `p.y >= 0`. The time step is adjusted so as to
// match the time skew of the simulation domain.
//
__dhce__ XYT WrapDomainPos(XYT p, RunShape rs) {
  return WrapY(WrapX(p, rs), rs);
}

// TODO: Document.
__dhce__ XYT StartLayerPos(UV b, RunShape rs) {
  return StartDomainPos(/*warppos=*/UV(0, 0), b, rs) +
         XYT(diamond::N / 2, Diamond(rs).u, 0);
}

} // namespace scanner

#endif // _SCANNER_IMPL_H_
