// Basic definitions.

#ifndef _SHAPEDEFS_H_
#define _SHAPEDEFS_H_

#include "macros.h"

namespace defs {

// `(x, y)` position in the simulation domain.
struct XY {
  __dhce__ XY(int x, int y) : x(x), y(y) {}
  int x, y;
};

// `(x, y, t)` position in the simulation domain.
struct XYT {
  __dhce__ XYT(int x, int y, int t) : x(x), y(y), t(t) {}
  int x, y, t;
};

// Used to denote position along the u- and v-diagonals.
//
// The u and v directions generally relate to `(x, y)` positions as
//   - `u = (1, 1)` and `v = (-1, 1)`,
//   - although `u = (-1, -1)` is often used when referring to the warps/blocks
//     in a block/grid.
//
struct UV {
  __dhce__ UV(int u, int v) : u(u), v(v) {}
  int u, v;
};

// Arithmetic ops on `UV`, `XY`, and `XYT` objects.
__dhce__ UV operator+(UV a, UV b) { return UV(a.u + b.u, a.v + b.v); }
__dhce__ UV operator+(int a, UV b) { return UV(a + b.u, a + b.v); }
__dhce__ UV operator+(UV a, int b) { return UV(a.u + b, a.v + b); }
__dhce__ XY operator+(XY a, XY b) { return XY(a.x + b.x, a.y + b.y); }
__dhce__ XY operator+(XY a, int b) { return XY(a.x + b, a.y + b); }
__dhce__ XYT operator+(XYT a, XYT b) {
  return XYT(a.x + b.x, a.y + b.y, a.t + b.t);
}
__dhce__ UV operator-(UV a, UV b) { return UV(a.u - b.u, a.v - b.v); }
__dhce__ UV operator-(UV a, int b) { return UV(a.u - b, a.v - b); }
__dhce__ UV operator-(int a, UV b) { return UV(a - b.u, a - b.v); }
__dhce__ UV operator*(int a, UV b) { return UV(a * b.u, a * b.v); }
__dhce__ UV operator*(UV a, int b) { return UV(a.u * b, a.v * b); }
__dhce__ UV operator*(UV a, UV b) { return UV(a.u * b.u, a.v * b.v); }
__dhce__ XY operator*(int a, XY b) { return XY(a * b.x, a * b.y); }
__dhce__ XYT operator*(int a, XYT b) { return XYT(a * b.x, a * b.y, a * b.t); }
__dhce__ UV operator%(UV a, UV b) { return UV(a.u % b.u, a.v % b.v); }
__dhce__ bool operator>(UV a, UV b) { return a.u > b.u && a.v > b.v; }
__dhce__ bool operator>=(UV a, UV b) { return a.u >= b.u && a.v >= b.v; }
__dhce__ int Sum(UV a) { return a.u + a.v; }
__dhce__ int Prod(UV a) { return a.u * a.v; }
__dhce__ int Prod(XY a) { return a.x * a.y; }
__dhce__ UV VU(UV a) { return UV(a.v, a.u); }

// Parameterizes the simulation kernel.
struct RunShape {
  struct Pml {
    __dhce__ Pml(int n, int zshift) : n(n), zshift(zshift) {}
    const int n;      // Number of threads in a warp for auxiliary PML values.
    const int zshift; // Number of cells to shift upward with wrap-around.
  };

  struct Src {
    enum Type {
      YSLICE, // `Ex` and `Ez` source at `y = y0`.
      ZSLICE  // `Ex` and `Ey` source at `z = z0`.
    };
    __dhce__ Src(Type type, int pos) : type(type), pos(pos) {}
    const Type type; // Type of source.
    const int pos;   // Position of source.
  };

  struct Out {
    __dhce__ Out(int start, int interval, int num)
        : start(start), interval(interval), num(num) {}
    const int start;    // Begin recording output at this timestep (inclusive).
    const int interval; // Number of timesteps between outputs.
    const int num;      // Number of outputs to write.
  };

  struct Vol {
    __dhce__ Vol(int x0, int x1, int y0, int y1, int z0, int z1)
        : x0(x0), x1(x1), y0(y0), y1(y1), z0(z0), z1(z1) {}
    const int x0, x1, y0, y1, z0, z1;
  };

  __dhce__ RunShape(                                                 //
      UV blockshape, UV gridshape, int blockspacing, XY domainshape, //
      Pml pml, Src src, Out out, Vol sub, Vol vol)
      : block(blockshape), grid(gridshape), spacing(blockspacing),
        domain(domainshape), pml(pml), src(src), out(out), sub(sub), vol(vol) {}

  // Helpful for testing.
  __dhce__ RunShape( //
      UV blockshape, UV gridshape, int blockspacing, XY domainshape)
      : RunShape(blockshape, gridshape, blockspacing, domainshape,
                 Pml(/*n=*/0, /*zshift=*/0), Src(Src::ZSLICE, /*pos=*/0),
                 Out(/*start=*/0, /*interval=*/0, /*num=*/0),
                 /*sub=*/Vol(0, 0, 0, 0, 0, 0),
                 /*vol=*/Vol(0, 0, 0, 0, 0, 0)) {}

  const UV block, grid; // Size of the diamond at the block and grid levels.
  const int spacing;    // Delay between adjacent blocks in the grid.
  const XY domain;      // Extent of the simulation domain.
  const Pml pml;        // Parameters related to the PML.
  const Src src;        // Parameters related to current sources.
  const Out out;        // Parameters related to output fields.
  const Vol sub, vol;   // Value ranges for subvolume and volume.
};

}; // namespace defs

#endif // _SHAPEDEFS_H_
