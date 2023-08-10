// Implements the arithmetic needed for the FDTD update.

#ifndef _UPDATE_H_
#define _UPDATE_H_

#include <cuda_fp16.h>

#include "defs.h"
#include "diamond.h"
#include "field.h"
#include "slice.h"
#include "zcoeff.h"

namespace update {

using defs::One;
using defs::RunShape;
using defs::UV;
using defs::XY;
using defs::XYT;
using defs::Zero;
using diamond::C;
using diamond::Cell;
using diamond::E;
using diamond::Ehc;
using diamond::H;
using diamond::N;
using diamond::Node;
using diamond::Nz;
using diamond::X;
using diamond::Xyz;
using diamond::Y;
using diamond::Z;
using zcoeff::ZCoeff;

// Used to pass values between neighboring threads along the z-axis.
//
// `npml` denotes the threads that are used to only hold auxiliary PML field
// values, specifically those with `ThreadId() >= kWarpSize - npml`. Each
// auxiliary thread is paired with a normal thread for which it stores auxiliary
// values -- communication occurs via `SwapShuffle()`.
//
template <typename T> __device__ T CircShuffle(T val, int offset, int npml) {
  const unsigned fullmask = 0xffffffff;
  int q = defs::kWarpSize - npml;
  int t = defs::ThreadPos();
  int src = t < q ? (t + q + offset) % q : t;
  return __shfl_sync(fullmask, val, src, defs::kWarpSize);
}

// Used to exchange values between corresponding field and auxiliary threads
// (needed for PML implementation along z-axis).
template <typename T> __device__ T SwapShuffle(T val) {
  const unsigned fullmask = 0xffffffff;
  return __shfl_xor_sync(fullmask, val, defs::kWarpSize - 1, defs::kWarpSize);
}

// Spatial derivative along x-axis.
template <typename T> __device__ T Dx(const Cell<T> &cell, Node n) {
  if (n.ehc == E)
    return cell.Get(n.dI(+1)) - cell.Get(n);
  else // n.ehc == H.
    return cell.Get(n) - cell.Get(n.dI(-1));
}

// Spatial derivative along y-axis.
template <typename T> __device__ T Dy(const Cell<T> &cell, Node n) {
  if (n.ehc == E)
    return cell.Get(n.dJ(+1)) - cell.Get(n);
  else // n.ehc == H.
    return cell.Get(n) - cell.Get(n.dJ(-1));
}

// Spatial derivative along z-axis.
template <typename T> __device__ T Dz(const Cell<T> &cell, Node n, int npml) {
  if (n.ehc == E)
    return (n.k < Nz - 1 ? cell.Get(n.dK(+1))
                         : CircShuffle(cell.Get(n.K(0)), +1, npml)) -
           cell.Get(n);
  else                   // n.ehc == H.
    return cell.Get(n) - //
           (n.k > 0 ? cell.Get(n.dK(-1))
                    : CircShuffle(cell.Get(n.K(Nz - 1)), -1, npml));
}

#ifndef __OMIT_HALF2__
// `half2` version takes into account the relationship between low and high
// `half2` values.
//
// Specifically, we use `Nz` low values arranged beneath `Nz` high values. This
// necessitates special finangling at this intersection within the cell and at
// the cell boundaries (these happen to basically be the "same place").
//
template <>
__device__ half2 Dz<half2>(const Cell<half2> &cell, Node n, int npml) {
  if (n.ehc == E) {
    return (n.k < Nz - 1 ? cell.Get(n.dK(+1))
                         : __halves2half2(__high2half(cell.Get(n.K(0))),
                                          __low2half(CircShuffle(
                                              cell.Get(n.K(0)), +1, npml)))) -
           cell.Get(n);
  } else {               // n.ehc == H.
    return cell.Get(n) - //
           (n.k > 0 ? cell.Get(n.dK(-1))
                    : __halves2half2(__high2half(CircShuffle(
                                         cell.Get(n.K(Nz - 1)), -1, npml)),
                                     __low2half(cell.Get(n.K(Nz - 1)))));
  }
}
#endif

// Implements the curl operation for both normal and auxiliary fields, which
// require that the correct `a` and `b` coefficients be loaded into `zcoeff`.
//
// Note that we include extraneous d{EH}z/dy and d{EH}z/dx components in the
// auxiliary updates (once again, to avoid branching which is lethal to
// performance) but we assume that these field values will always be zero, since
// they are not needed as auxiliary values (PML only along z-axis).
//
template <typename T>
__device__ T Curl(const Cell<T> &cell, Node node, ZCoeff<T> zcoeff, int npml,
                  bool isaux) {
  Node n = node.ehc == E ? node.dI(+1) : node;
  T a = zcoeff.Get(n.k, n.ehc, zcoeff::InternalType::A);
  T b = zcoeff.Get(n.k, n.ehc, zcoeff::InternalType::B);

  if (n.xyz == X) {
    T dz = Dz(cell, n.Dual(Y), npml);
    // The `-` sign is used here because we actually store the negative
    // auxiliary values for Ex and Hx components in order to avoid branching at
    // the warp level.
    dz = a * dz + SwapShuffle((isaux ? -cell.Get(node) : b * dz));
    return Dy(cell, n.Dual(Z)) - dz;

  } else if (n.xyz == Y) {
    T dz = Dz(cell, n.Dual(X), npml);
    dz = a * dz + SwapShuffle((isaux ? cell.Get(node) : b * dz));
    return dz - Dx(cell, n.Dual(Z));

  } else { // n.xyz == Z
    return Dx(cell, n.Dual(Y)) - Dy(cell, n.Dual(X));
  }
}

// Implements a stripped-down version of the FDTD update which does not include
// sources or the absorption mask which must be implemented elsewhere.
template <typename T>
__device__ void Update(Cell<T> &cell, Node n, ZCoeff<T> zcoeff, const T d,
                       int p, bool isaux) {
  T val;
  if (n.ehc == E) {
    // Only the E-field update requires the material coefficients.
    //
    // Note that the c-value for auxiliary threads must be set to `1`.
    //
    val = cell.Get(n) + cell.Get(n.AsC()) * Curl(cell, n, zcoeff, p, isaux);
  } else { // n.ehc == H.
    val = cell.Get(n) + d * Curl(cell, n, zcoeff, p, isaux);
  }

  // Don't update auxiliary Ez/Hz fields.
  if (!isaux || n.xyz != Z)
    cell.Set(val, n);
}

// Update all `ehc` nodes in a cell.
template <typename T>
__device__ void Update(Cell<T> &cell, ZCoeff<T> zcoeff, const T d, int p,
                       bool isaux, Ehc ehc) {

#pragma unroll
  for (int i : diamond::AllI)
#pragma unroll
    for (int j : diamond::AllJ)
#pragma unroll
      for (int k : diamond::AllK)
#pragma unroll
        for (diamond::Xyz xyz : diamond::AllXyz) {
          diamond::Node n(i, j, k, ehc, xyz);
          if (diamond::IsInsideDiamond(n))
            Update(cell, n, zcoeff, d, p, isaux);
        }
}

// Scale the E- or H-field nodes of a cell by the correct coefficient.
//
// This is implmemented separate from `Update()` to avoid performance cliffs
// associated with the `nvcc` compiler.
//
template <typename T>
__device__ void Scale(Cell<T> &cell, slice::ZMask<T> zmask, ZCoeff<T> zcoeff,
                      bool isaux, Ehc ehc) {
  int cnt = 0;
#pragma unroll
  for (int i : diamond::AllI)
#pragma unroll
    for (int j : diamond::AllJ)
#pragma unroll
      for (diamond::Xyz xyz : diamond::AllXyz) {
        Node n0(i, j, /*k=*/0, ehc, xyz);
        if (diamond::IsInsideDiamond(n0)) {
          // The absorption mask is only needed for the E-field.
          T abs = (ehc == E ? zmask.Get(cnt) : One<T>());
          ++cnt;
#pragma unroll
          for (int k : diamond::AllK) {
            Node n = n0.K(k);
            cell.Set(
                (isaux ? zcoeff.Get(n.k, ehc, zcoeff::InternalType::B) : abs) *
                    cell.Get(n),
                n);
          }
        }
      }
}

// Implement the z-plane source.
template <typename T>
__device__ void ZPlaneSrc(Cell<T> &cell, const slice::ZSrc<T> &zsrc,
                          const T srccoeff[2], RunShape rs) {
  // slice::ZSrc<T> zsrc;
  // zsrc.Load(ptr, pos, rs.domain, threadpos);
  int cnt = 0;
#pragma unroll
  for (int i : diamond::AllI)
#pragma unroll
    for (int j : diamond::AllJ)
#pragma unroll
      for (diamond::Xyz xyz : {X, Y}) {
        Node n0(i, j, /*k=*/0, E, xyz);
        if (diamond::IsInsideDiamond(n0)) {
          T srcval =
              srccoeff[0] * zsrc.Get(cnt) + srccoeff[1] * zsrc.Get(cnt + 1);
          cnt += 2; // Needed to match the ordering of nodes in `ZSrc`.
#pragma unroll
          for (int k : diamond::AllK) {
            Node n = n0.K(k);
            cell.Set(
                cell.Get(n) + (rs.src.pos % Nz == n.k ? srcval : Zero<T>()), n);
          }
        }
      }
}

// Implements the y-plane source.
template <typename T>
__dhce__ void YPlaneSrc(Cell<T> &cell, const slice::YSrc<T> &ysrc,
                        const T srccoeff[2], int srcpos, XY pos) {
  // `srcpos` is assumed to be even for simplicity.
  if (pos.y == srcpos) {
#pragma unroll
    for (int k : diamond::AllK) {
#pragma unroll
      for (int i : {-2, -1, 0}) {
        {
          Node n(i, 0, k, E, X);
          cell.Set(cell.Get(n) + srccoeff[0] * ysrc.Get(-i, k, X), n);
        }
        {
          Node n(i + 1, 0, k, E, Z);
          cell.Set(cell.Get(n) + srccoeff[0] * ysrc.Get(-i, k, Z), n);
        }
      }
      {
        Node n(-1, -1, k, E, X);
        cell.Set(cell.Get(n) + srccoeff[1] * ysrc.Get(1, k, X), n);
      }
      {
        Node n(0, -1, k, E, Z);
        cell.Set(cell.Get(n) + srccoeff[1] * ysrc.Get(1, k, Z), n);
      }
    }
  } else if (pos.y == srcpos - 2) {
#pragma unroll
    for (int k : diamond::AllK) {
      {
        Node n(-1, 2, k, E, X);
        cell.Set(cell.Get(n) + srccoeff[0] * ysrc.Get(1, k, X), n);
      }
      {
        Node n(0, 2, k, E, Z);
        cell.Set(cell.Get(n) + srccoeff[0] * ysrc.Get(1, k, Z), n);
      }
#pragma unroll
      for (int i : {-2, -1, 0}) {
        {
          Node n(i, 1, k, E, X);
          cell.Set(cell.Get(n) + srccoeff[1] * ysrc.Get(-i, k, X), n);
        }
        {
          Node n(i + 1, 1, k, E, Z);
          cell.Set(cell.Get(n) + srccoeff[1] * ysrc.Get(-i, k, Z), n);
        }
      }
    }
  }
}

// Adds the source term to the E-field update.
template <typename T, typename T1>
__device__ void AddSrc(Cell<T> &cell, const slice::ZSrc<T> &zsrc,
                       const slice::YSrc<T> &ysrc, const T1 *wf, XYT domainpos,
                       RunShape rs, int t) {
  T srccoeff[2];
  if (rs.src.type == RunShape::Src::ZSLICE) {
    if (t == rs.src.pos / diamond::EffNz<T>() && //
        domainpos.t >= 0 && domainpos.t < defs::NumTimeSteps(rs.out)) {
      srccoeff[0] = defs::Convert<T, T1>(wf[2 * domainpos.t]);
      srccoeff[1] = defs::Convert<T, T1>(wf[2 * domainpos.t + 1]);
    } else {
      srccoeff[0] = defs::Zero<T>();
      srccoeff[1] = defs::Zero<T>();
    }
    update::ZPlaneSrc(cell, zsrc, srccoeff, rs);

  } else { // rs.srctype == RunShape::Src::YSLICE.
    if (domainpos.t >= 0 && domainpos.t < defs::NumTimeSteps(rs.out)) {
      srccoeff[0] = defs::Convert<T, T1>(wf[2 * domainpos.t]);
      srccoeff[1] = defs::Convert<T, T1>(wf[2 * domainpos.t + 1]);
    } else {
      srccoeff[0] = defs::Zero<T>();
      srccoeff[1] = defs::Zero<T>();
    }
    update::YPlaneSrc(cell, ysrc, srccoeff, rs.src.pos,
                      XY(domainpos.y, domainpos.y));
  }
}

// Writes the nodes in the diamond to the output buffer.
template <typename T, typename T1>
__device__ void WriteOutput(Cell<T> &cell, XYT domainpos, int threadpos,
                            T1 *outptr, int zshift, RunShape rs, bool isaux) {
  if (
      // diamond::IsDiamondCompletelyInDomain(XY(domainpos.x, domainpos.y),
      //                                      rs.domain) &&
      diamond::IsDiamondCompletelyInXY(XY(domainpos.x, domainpos.y),
                                       rs.sub.x0 - N, rs.sub.x1 + N,
                                       rs.sub.y0 - N, rs.sub.y1 + N) &&
      domainpos.t >= rs.out.start &&                               //
      domainpos.t < rs.out.start + rs.out.num * rs.out.interval && //
      (domainpos.t - rs.out.start) % rs.out.interval == 0) {
    int outindex = (domainpos.t - rs.out.start) / rs.out.interval;
    field::WriteCell(cell, outptr, XY(domainpos.x, domainpos.y), outindex,
                     threadpos, rs.domain, rs.pml.n, zshift, isaux, rs.sub);
  }
}

} // namespace update

#endif // _UPDATE_H_
