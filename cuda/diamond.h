// Defines the diamond cell assigned to each thread.

#ifndef _DIAMOND_H_
#define _DIAMOND_H_

#include "defs.h"

namespace diamond {

using defs::kWarpSize;
using defs::RunShape;
using defs::XY;

// Length of the diagonal of the FDTD diamond in the x-y plane in terms of Yee
// cells. A `N`-sized diamond contains `N * N / 2` Yee cells in the x-y plane.
constexpr const int N = 4;

// Number of `N`-sized diamond layers along the `z`-axis per thread.
constexpr const int Nz = 2;

// "Effective" Nz, since each `half2` value actually contains 2 layers.
template <typename T> __dhce__ int EffNz() { return Nz; }

#ifndef __OMIT_HALF2__
template <> __dhce__ int EffNz<half2>() { return 2 * Nz; }
#endif

// External number of layers in the simulation domain.
template <typename T> __dhce__ int ExtZz(int npml) {
  return EffNz<T>() * (kWarpSize - npml);
}

// Positive modulo, assumes `y > 0`.
__dhce__ int PosMod(int x, int y) { return ((x % y) + y) % y; }

// External z-index.
template <typename T>
__dhce__ int ExtZIndex(int k, int threadpos, int npml, int zshift) {
  // First bring auxiliary cells into the external z-domain.
  if (defs::IsAux(threadpos, npml))
    threadpos = kWarpSize - threadpos - 1;

  return PosMod((k + EffNz<T>() * threadpos) - zshift, ExtZz<T>(npml));
}

// Vector-field types and components
//
// E, H, and C three-dimensional vector fields are supported for electric-,
// magnetic-, and conefficient-fields (used in the E-field update) respectively.
//
enum Ehc { E, H, C };
enum Xyz { X, Y, Z };
constexpr const int kNumEhc = 3;
constexpr const int kNumXyz = 3;

__dhce__ int Index(Ehc ehc) {
  if (ehc == E)
    return 0;
  else if (ehc == H)
    return 1;
  else // ehc == C.
    return 2;
}

__dhce__ int Index(Xyz xyz) {
  if (xyz == X)
    return 0;
  else if (xyz == Y)
    return 1;
  else // xyz == Z.
    return 2;
}

// Makes performant loops a little less verbose.
__device__ constexpr const int AllI[6] = {-2, -1, 0, 1, 2, 3};
__device__ constexpr const int AllJ[5] = {-2, -1, 0, 1, 2};
__device__ constexpr const int AllK[2] = {0, 1};
__device__ constexpr const Ehc AllEhc[kNumEhc] = {E, H, C};
__device__ constexpr const Xyz AllXyz[kNumXyz] = {X, Y, Z};

// `(i, j, k, ehc, xyz)` designation of a node.
// TODO: Better documentation.
// TODO: Consider separate IJK object.
struct Node {
  __dhce__ Node(int i, int j, int k, Ehc ehc, Xyz xyz)
      : i(i), j(j), k(k), ehc(ehc), xyz(xyz) {}
  const int i, j, k;
  const Ehc ehc;
  const Xyz xyz;

  __dhce__ Node I(int val) const { return Node(val, j, k, ehc, xyz); }
  __dhce__ Node J(int val) const { return Node(i, val, k, ehc, xyz); }
  __dhce__ Node K(int val) const { return Node(i, j, val, ehc, xyz); }
  __dhce__ Node dI(int d) const { return Node(i + d, j, k, ehc, xyz); }
  __dhce__ Node dJ(int d) const { return Node(i, j + d, k, ehc, xyz); }
  __dhce__ Node dK(int d) const { return Node(i, j, k + d, ehc, xyz); }
  __dhce__ Node As(Ehc asehc, Xyz asxyz) const {
    return Node(i, j, k, asehc, asxyz);
  }
  __dhce__ Node AsE() const { return As(E, xyz); }
  __dhce__ Node AsH() const { return As(H, xyz); }
  __dhce__ Node AsC() const { return As(C, xyz); }
  __dhce__ Node AsEx() const { return As(E, X); }
  __dhce__ Node AsEy() const { return As(E, Y); }
  __dhce__ Node AsEz() const { return As(E, Z); }
  __dhce__ Node AsHx() const { return As(H, X); }
  __dhce__ Node AsHy() const { return As(H, Y); }
  __dhce__ Node AsHz() const { return As(H, Z); }
  __dhce__ Node AsCx() const { return As(C, X); }
  __dhce__ Node AsCy() const { return As(C, Y); }
  __dhce__ Node AsCz() const { return As(C, Z); }
  __dhce__ Node Dual(Xyz xyz) const { return As(ehc == E ? H : E, xyz); }
  __dhce__ Node Shift(int delta, Xyz dir) const {
    if (dir == X)
      return Node(i + delta, j, k, ehc, xyz);
    else if (dir == Y)
      return Node(i, j + delta, k, ehc, xyz);
    else // dir == Z
      return Node(i, j, k + delta, ehc, xyz);
  }
};

// Returns `true` if the node at `(i, j, c)` is inside the diamond.
//
// Uses the conventional Yee cell with offsets of
// - `(0.5, 0)` for Ex,
// - `(0, 0.5)` for Ey,
// - `(0, 0)` for Ez,
// - `(0, 0.5)` for Hx,
// - `(0.5, 0)` for Hy, and
// - `(0.5, 0.5)` for Hz.
//
// Adopts the convention that the diamond is centered at
// - `(-0.25, 0.5)` for E-fields, and
// - `(0.25, 0.5)` for H-fields.
// that is, the H-diamond is shifted a half-cell in `+x`.
//
// Note that C-fields are co-located with E-fields.
//
__dhce__ bool IsInsideDiamond(Node n) {
  float x = ((n.ehc == H) != (n.xyz == X)) ? n.i + 0.5 : n.i;
  float y = ((n.ehc == H) != (n.xyz == Y)) ? n.j + 0.5 : n.j;
  float cx = (n.ehc == H) ? 0.25 : -0.25;
  float cy = 0.5;
  return ((abs(x - cx) + abs(y - cy)) < (N / 2.0)) && (n.k >= 0 && n.k < Nz);
}

// Returns `true` iff `(i, j, c)` is in the trailing edge of the diamond.
//
// Trailing edge nodes are defined as those within the diamond and do not have
// a neighboring node in the diamond in the -x direction.
//
__dhce__ bool IsTrailingEdge(Node n) {
  return IsInsideDiamond(n) && !IsInsideDiamond(n.dI(-1));
}

// Returns `true` iff `(i, j, c)` is in the leading edge of the diamond.
//
// Nodes in the leading edge are defined as
// - not being in the diamond, and either
//   - being in the diamond that is shifted by a cell in the +x direction, or
//   - being a dependency of the dual field diamond, where the dual E-field is
//     shifted by a cell in the +x direction when computing the H-field
//     leading edge nodes.
//
__dhce__ bool IsLeadingEdge(Node n) {
  if (n.ehc == E) {
    if (n.xyz == X)
      return !IsInsideDiamond(n) &&        //
             (IsInsideDiamond(n.dI(-1)) || //
              IsInsideDiamond(n.AsHy()) || //
              IsInsideDiamond(n.AsHz()) || //
              IsInsideDiamond(n.AsHz().dJ(-1)));
    else if (n.xyz == Y)
      return !IsInsideDiamond(n) &&        //
             (IsInsideDiamond(n.dI(-1)) || //
              IsInsideDiamond(n.AsHx()) || //
              IsInsideDiamond(n.AsHz()) || //
              IsInsideDiamond(n.AsHz().dI(-1)));
    else                                          // n.xyz == Z.
      return !IsInsideDiamond(n) &&               //
             (IsInsideDiamond(n.dI(-1)) ||        //
              IsInsideDiamond(n.AsHx()) ||        //
              IsInsideDiamond(n.AsHx().dJ(-1)) || //
              IsInsideDiamond(n.AsHy()) ||        //
              IsInsideDiamond(n.AsHy().dI(-1)));

  } else if (n.ehc == H) {
    if (n.xyz == X)
      return !IsInsideDiamond(n) &&               //
             (IsInsideDiamond(n.dI(-1)) ||        //
              IsInsideDiamond(n.dI(-1).AsEy()) || //
              IsInsideDiamond(n.dI(-1).AsEz()) || //
              IsInsideDiamond(n.dI(-1).AsEz().dJ(+1)));
    else if (n.xyz == Y)
      return !IsInsideDiamond(n) &&               //
             (IsInsideDiamond(n.dI(-1)) ||        //
              IsInsideDiamond(n.dI(-1).AsEx()) || //
              IsInsideDiamond(n.dI(-1).AsEz()) || //
              IsInsideDiamond(n.AsEz()));
    else                                                 // n.xyz == Z.
      return !IsInsideDiamond(n) &&                      //
             (IsInsideDiamond(n.dI(-1)) ||               //
              IsInsideDiamond(n.dI(-1).AsEx()) ||        //
              IsInsideDiamond(n.dI(-1).AsEx().dJ(+1)) || //
              IsInsideDiamond(n.dI(-1).AsEy()) ||        //
              IsInsideDiamond(n.AsEy()));

  } else { // n.ehc == C.
    // Leading edge C nodes are those which do not have a neighbor in the +x
    // direction.
    return IsInsideDiamond(n) && !IsInsideDiamond(n.dI(+1));
  }
}

// Top and bottom Ey point which needs to be included for buffering.
__dhce__ bool IsTopBotEy(Node n) {
  return (n.i == 0) &&                      //
         (n.j == -N / 2 || n.j == N / 2) && //
         (n.k >= 0 && n.k < Nz) &&          //
         n.ehc == E &&                      //
         (n.xyz == Y);
}

// `true` is `n` may be needed.
__dhce__ bool IsActive(Node n) {
  return IsInsideDiamond(n) || IsLeadingEdge(n) || IsTopBotEy(n);
}

// TODO: Document and test.
__dhce__ bool IsDiamondCompletelyInDomain(XY pos, XY domain) {
  return pos.x >= (diamond::N / 2) &&                 //
         pos.x < domain.x - ((diamond::N / 2) - 1) && //
         pos.y >= diamond::N / 2 &&                   //
         pos.y < domain.y - (diamond::N / 2);
}

// TODO: Document and test.
__dhce__ bool IsDiamondCompletelyInXY(XY pos, int x0, int x1, int y0, int y1) {
  return pos.x >= x0 + (diamond::N / 2) &&      //
         pos.x < x1 - ((diamond::N / 2) - 1) && //
         pos.y >= y0 + (diamond::N / 2) &&      //
         pos.y < y1 - (diamond::N / 2);
}

// range-based for loop
//
// So that we can do something like
//
// for (Node n : AllNodes)
//   if (IsInsideDiamond(n))
//     ...
//
// Local (register) storage and "0-centered" access for the nodes of a diamond.
//
// Note: In use, `Cell` objects should be passed by reference or constant
// reference -- the nvidia compiler will often take a very long time to compile
// if it is passed by value (it seems to take a long time to figure out that the
// copy is not needed).
//
template <typename T> struct Cell {

  template <typename T1 = T> __dh__ T1 Get(Node n) const {
    return ((T1 *)(values_[n.i + N / 2][n.j + N / 2][n.k][Index(n.ehc)]))[Index(
        n.xyz)];
  }

  template <typename T1 = T> __dh__ void Set(T1 val, Node n) {
    ((T1 *)(values_[n.i + N / 2][n.j + N / 2][n.k][Index(n.ehc)]))[Index(
        n.xyz)] = val;
  }

private:
  // Note that `N + 2` values are needed in the x-direction since the diamond
  // nodes extend from `-N / 2` to `N / 2 + 1`, while `N + 1` values are needed
  // in the y-direction since it needs to extend from `-N / 2` to `N / 2` only.
  // The extra value is needed along the x-axis because of the leading
  // edge/point in that direction.
  T values_[N + 2][N + 1][Nz][kNumEhc][kNumXyz];
};

// Returns an cell with all active nodes initialized to `value`.
template <typename T> __dhce__ void InitCell(Cell<T> &cell, T value) {
#pragma unroll
  for (int i : AllI)
#pragma unroll
    for (int j : AllJ)
#pragma unroll
      for (int k : AllK)
#pragma unroll
        for (Ehc ehc : AllEhc)
#pragma unroll
          for (Xyz xyz : AllXyz) {
            Node n(i, j, k, ehc, xyz);
            if (IsActive(n))
              cell.Set(value, n);
          }
}

template <typename T> __dh__ void Shift(Cell<T> &cell, Ehc ehc) {
#pragma unroll
  for (int i : AllI)
#pragma unroll
    for (int j : AllJ)
#pragma unroll
      for (int k : AllK)
#pragma unroll
        for (Xyz xyz : AllXyz) {
          Node n(i, j, k, ehc, xyz);
          if (IsInsideDiamond(n))
            cell.Set(cell.Get(n.dI(+1)), n);
        }
}

} // namespace diamond

#endif // _DIAMOND_H_
