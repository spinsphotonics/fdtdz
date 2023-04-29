// Coordinates how each diamond traverses the simulation domain.

#ifndef _SCANNER_H_
#define _SCANNER_H_

#include "defs.h"
#include "diamond.h"
#include "scanner_impl.h"

namespace scanner {

using defs::RunShape;
using defs::UV;
using defs::XY;
using defs::XYT;

// Buffer shape corresponding to domain and diamond shape.
//
// Scanning advances in the +x direction at every step and the +y direction at
// the end of every scan line. The u-buffer is iterated through once for every
// scan of the entire domain, while the v-buffer is iterated through on every
// scan line.
//
__dhce__ UV BufferShape(RunShape rs) {
  UV d = Diamond(rs);
  return UV(((rs.domain.y - d.u) / d.v) * rs.domain.x - d.u, rs.domain.x - d.v);
}

// In order to be valid, the `RunShape` must be compatible with both
// the time-skew of the simulation domain and the buffering scheme.
//
// Compatibility with the buffering scheme requires that the buffer shape be
// large enough to accomodate the inter-block spacing and that the y-extent of
// the domain accomodate exactly one diamond u-extent and one or more diamond
// v-extents.
//
// Compatibility with the time-skew requires that the v-extent of the diamond be
// equal to or larger than the u-extent. This is related to the skew being from
// `(0, 0) -> (xx, yy)` in the simulation domain.
//
__host__ bool IsValidRunShape(RunShape rs) {
  return BufferShape(rs) >= rs.grid * rs.spacing &&
         (rs.domain.y - Diamond(rs).u) % Diamond(rs).v == 0 &&
         (rs.domain.y - Diamond(rs).u) / Diamond(rs).v > 0 &&
         Diamond(rs).v >= Diamond(rs).u;
}

// Number of steps needed to guarantee that every node of the grid has reached
// `timestep` or beyond.
//
// Uses the fact that
// 1. a scan requires `domain.x * (domain.y / diamond.v)` steps,
// 2. a scan advances each node by `2 * diamond.u` timesteps, and
// 3. the rear-most point `(0, 0)` starts at timestep `-(domain.x + domain.y)`
//    at worst.
//
// TODO: Add unit tests.
__dhce__ int NumSteps(int timesteps, RunShape rs) {
  // Actual number of passes needed per node, taking into account both the
  // skewed nature of the domain and that each pass advances a node by `2 *
  // Diamond(rs).u` steps.
  int numpasses =
      ((timesteps + rs.domain.x + rs.domain.y) / (2 * Diamond(rs).u)) + 1;

  // Number of scan lines needed, since each scan performs updates for
  // `Diamond(rs).v` rows of nodes.
  int numlines = ((rs.domain.y * numpasses) / Diamond(rs).v) + 1;

  // Each scan line needs `rs.domain.x` steps.
  return rs.domain.x * numlines;

  //
  //       // Number of line scanes needed to
  //       int numlines =
  //           (skewsteps * rs.domain.y / (2 * Diamond(rs).u)) + 1 int numlines
  //           =
  //               (rs.domain.y * (timesteps + rs.domain.x + rs.domain.y)) /
  //                   (2 * Diamond(rs).u) +
  //               1;
  //   return rs.domain.x * numlines;

  // int stepsperscan = rs.domain.x * (rs.domain.y / Diamond(rs).v + 0);
  // int numscans =
  //     (timesteps + rs.domain.x + rs.domain.y) / (2 * Diamond(rs).u) + 1;
  // return numscans * stepsperscan;
  // return (timesteps + rs.domain.x + rs.domain.y) *
  //        (rs.domain.x * rs.domain.y / (2 * Diamond(rs).u * Diamond(rs).v));
}

// The position of the diamond in both u- and v-buffers.
//
// The u- and v-buffers are implemented as wrap-around buffers for inter-block
// communication.
//
// TODO: Try incremental version, which may use fewer registers.
__dhce__ UV BufferPos(int step, UV blockpos, RunShape rs) {
  return ((rs.grid - blockpos - 1) * rs.spacing + step) % BufferShape(rs);
}

// The position of the diamond in the simulation domain at `step`.
//
// Specifically, this gives the spatial (and temporal) offset of the diamond
// nodes within the spatial domain.
//
// Implements a scheme where the diamond is allowed to continually scan without
// interruption, by wrapping around the x- and y-boundaries of the simulation.
// Since this allows a single diamond to straddle a boundary, the original
// simulation domain needs to be padded by at least `diamond::N / 2` cells at
// each xy-boundary.
//
__dhce__ XYT DomainPos(int step, UV warppos, UV blockpos, RunShape rs) {
  // With every step, the diamond is advanced along the x-axis as well as in
  // time. The resulting position and time step is then obtained by wrapping
  // within the simulation domain.
  return WrapDomainPos(
      StartDomainPos(warppos, blockpos, rs) + XYT(step, 0, step), rs);
}

} // namespace scanner

#endif // _SCANNER_H_
