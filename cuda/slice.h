// Data structures and functions for 2D (slice) data structures.

#ifndef _SLICE_H_
#define _SLICE_H_

#include "defs.h"
#include "diamond.h"

namespace slice {

using defs::kWarpSize;
using defs::RunShape;
using defs::UV;
using defs::XY;
using diamond::Cell;
using diamond::E;
using diamond::EffNz;
using diamond::Ehc;
using diamond::ExtZIndex;
using diamond::ExtZz;
using diamond::H;
using diamond::N;
using diamond::Node;
using diamond::Nz;
using diamond::Xyz;

// Simple structure that features exactly `kWarpSize` elements per domain
// position. Takes advantage of the fact that domain position along y-axis must
// always be even.
template <typename T> struct Layer {
  __dh__ Layer() : val(defs::Zero<T>()) {}

  __dhsc__ int GlobalElems(XY domain) {
    return domain.x * (domain.y / 2) * kWarpSize;
  }

  __dhsc__ int GlobalIndex(int index, XY pos, XY domain) {
    return index + kWarpSize * (pos.x + domain.x * (pos.y / 2));
  }

  __dh__ void Load(T *ptr, XY pos, XY domain, int threadpos) {
    val = ptr[GlobalIndex(threadpos, pos, domain)];
  }

  __device__ T CastShuffle(T val, int i) const {
    const unsigned fullmask = 0xffffffff;
    return __shfl_sync(fullmask, val, i, defs::kWarpSize);
  }

  __dh__ T Get(int i) const { return CastShuffle(val, i); }

  T val;
};

// Absorption mask.
template <typename T> struct ZMask : public Layer<T> {
  __dh__ ZMask() : Layer<T>() {}

  __dhsc__ int ExternalElems(XY domain) {
    return defs::Prod(domain) * diamond::kNumXyz;
  }

  __dhsc__ int ExternalIndex(XY pos, Xyz xyz, XY domain) {
    return pos.x + domain.x * (pos.y + domain.y * diamond::Index(xyz));
  }

  template <typename T1> __dhsc__ T ConvertToCoeff(T1 abs, T1 dt) {
    // return defs::Convert<T, T1>(abs);
    return defs::Convert<T, T1>(((1 / dt) - (abs / 2)) /
                                ((1 / dt) + (abs / 2)));
  }

  template <typename T1>
  __dhsc__ void WriteGlobal(T1 *src, T *dst, XY pos, XY domain, T1 dt) {
    int cnt = 0;
    for (int i : diamond::AllI)
#pragma unroll
      for (int j : diamond::AllJ)
#pragma unroll
        for (diamond::Xyz xyz : diamond::AllXyz)
          if (diamond::IsInsideDiamond(diamond::Node(i, j, /*k=*/0, E, xyz))) {
            XY p = pos + XY(i, j);
            bool isinside =
                p.x >= 0 && p.y >= 0 && p.x < domain.x && p.y < domain.y;
            dst[Layer<T>::GlobalIndex(cnt, pos, domain)] =
                isinside
                    ? ConvertToCoeff(src[ExternalIndex(p, xyz, domain)], dt)
                    : defs::Zero<T>();
            ++cnt;
          }
  }
};

// Used to convert a `float` to only either the low or high value of a `half2`.
template <typename T, typename T1> __dhce__ T SmartConvert(T1 val, int srcpos) {
  return val;
}

#ifndef __OMIT_HALF2__
template <> half2 SmartConvert(float val, int srcpos) {
  if (((srcpos / Nz) % 2) == 0)
    return __floats2half2_rn(val, defs::Zero<float>());
  else
    return __floats2half2_rn(defs::Zero<float>(), val);
}
#endif

// Current source as a plane along the z-axis.
template <typename T> struct ZSrc : public Layer<T> {
  static constexpr const int NumChannels = 2;
  static constexpr const int NumComps = 2;
  static constexpr const Xyz Comps[NumComps] = {diamond::X, diamond::Y};

  __dh__ ZSrc() : Layer<T>() {}

  __dhsc__ int ExternalElems(XY domain) {
    return defs::Prod(domain) * NumChannels * NumComps;
  }

  __dhsc__ int XyzIndex(Xyz xyz) {
    if (xyz == diamond::X)
      return 0;
    else // xyz == diamond::Y.
      return 1;
  }

  __dhsc__ int ExternalIndex(XY pos, Xyz xyz, int channel, XY domain) {
    return XyzIndex(xyz) +
           NumComps * (pos.x + domain.x * (pos.y + domain.y * channel));
  }

  // TODO: Test srcpos.
  template <typename T1>
  __dhsc__ void WriteGlobal(T1 *src, T *dst, int srcpos, XY pos, XY domain) {
    int cnt = 0;
    for (int i : diamond::AllI)
#pragma unroll
      for (int j : diamond::AllJ)
#pragma unroll
        for (diamond::Xyz xyz : {diamond::X, diamond::Y})
          if (diamond::IsInsideDiamond(diamond::Node(i, j, /*k=*/0, E, xyz))) {
            XY p = pos + XY(i, j);
            bool isinside =
                p.x >= 0 && p.y >= 0 && p.x < domain.x && p.y < domain.y;
            for (int channel = 0; channel < NumChannels; ++channel) {
              dst[Layer<T>::GlobalIndex(cnt, pos, domain)] =
                  isinside
                      ? SmartConvert<T, T1>(
                            src[ExternalIndex(p, xyz, channel, domain)], srcpos)
                      : defs::Zero<T>();
              ++cnt;
            }
          }
  }
};

// Current source for a slice in `y = y0`.
template <typename T> struct YSrc {
  static constexpr const int ChunkSize = 2 * Nz * kWarpSize;
  static constexpr const int NumComps = 2;

  __dhsc__ int XyzIndex(Xyz xyz) {
    if (xyz == diamond::X)
      return 0;
    else // xyz == diamond::Z.
      return 1;
  }

  __dhsc__ int ExternalElems(int xx, int npml) {
    return xx * ExtZz<T>(npml) * NumComps;
  }

  __dhsc__ int ExternalIndex(int i, int k, Xyz xyz, int npml) {
    return k + ExtZz<T>(npml) * (XyzIndex(xyz) + NumComps * i);
  }

  __dhsc__ int GlobalElems(int xx) { return ChunkSize * xx; }

  __dhsc__ int GlobalIndex(int i, int k, Xyz xyz, int threadpos, int xx) {
    // Rotate since Ez components are ahead of Ex in diamond.
    if (xyz == diamond::Z)
      i = (i - 1 + xx) % xx;
    return k + Nz * (XyzIndex(xyz) + NumComps * (threadpos + kWarpSize * i));
  }

  template <typename T1>
  __dhsc__ void WriteNode(T1 *src, T *dst, int i, int k, Xyz xyz, int xx,
                          int threadpos, int npml, int zshift, bool isaux) {
    int extz = ExtZIndex<T>(k, threadpos, npml, zshift);
    dst[GlobalIndex(i, k, xyz, threadpos, xx)] =
        isaux ? defs::Zero<T>() : src[ExternalIndex(i, extz, xyz, npml)];
  }

#ifndef __OMIT_HALF2__
  __dhsc__ void WriteNode(float *src, half2 *dst, int i, int k, Xyz xyz, int xx,
                          int threadpos, int npml, int zshift, bool isaux) {
    int extzlo = ExtZIndex<half2>(k, threadpos, npml, zshift);
    int extzhi = ExtZIndex<half2>(k + Nz, threadpos, npml, zshift);
    dst[GlobalIndex(i, k, xyz, threadpos, xx)] =
        isaux ? defs::Zero<half2>()
              : __floats2half2_rn(src[ExternalIndex(i, extzlo, xyz, npml)],
                                  src[ExternalIndex(i, extzhi, xyz, npml)]);
  }
#endif

  template <typename T1>
  __dhsc__ void WriteGlobal(T1 *src, T *dst, int i, int threadpos, int xx,
                            int npml, int zshift, bool isaux) {
#pragma unroll
    for (int k = 0; k < Nz; ++k)
#pragma unroll
      for (Xyz xyz : {diamond::X, diamond::Z})
        WriteNode(src, dst, i, k, xyz, xx, threadpos, npml, zshift, isaux);
  }

  __dh__ void Load(T *ptr, int x, int threadpos) {
    val[2] = val[1];
    val[1] = val[0];
    val[0] = ((uint4 *)ptr)[threadpos + kWarpSize * x];
  }

  __dhsc__ int ComponentIndex(int k, Xyz xyz) {
    if (k == 0 && xyz == diamond::X)
      return 0;
    else if (k == 1 && xyz == diamond::X)
      return 1;
    else if (k == 0 && xyz == diamond::Z)
      return 2;
    else // k == 1 && xyz == diamond::Z.
      return 3;
  }

  // Here, `i` refers to `i` steps (along x-axis) behind the current position.
  __dh__ T Get(int i, int k, Xyz xyz) const {
    return ((T *)val)[4 * i + ComponentIndex(k, xyz)];
  }

  __dh__ YSrc() {
    for (int i = 0; i < 3; ++i)
      val[i] = make_uint4(0, 0, 0, 0);
  }

  uint4 val[3];
};

// Convert between external and global buffer for the absorption mask.
template <typename T, typename T1>
__dh__ void ConvertMask(T1 *src, T *dst, RunShape rs, int threadpos, UV warppos,
                        UV blockpos, T1 dt) {
  UV init = warppos + rs.block * blockpos;
  UV stride = rs.block * rs.grid;
  for (int i = init.u; i < rs.domain.x; i += stride.u)
    for (int j = 2 * init.v; j < rs.domain.y;
         j += 2 * stride.v) // Only need even y positions.
      if (threadpos == 0)
        slice::ZMask<T>::WriteGlobal(src, dst, XY(i, j), rs.domain, dt);
}

// Convert between external and global buffer for the z-plane source.
template <typename T, typename T1>
__dh__ void ConvertZSrc(T1 *src, T *dst, int srcpos, RunShape rs, int threadpos,
                        UV warppos, UV blockpos) {
  UV init = warppos + rs.block * blockpos;
  UV stride = rs.block * rs.grid;
  for (int i = init.u; i < rs.domain.x; i += stride.u)
    for (int j = 2 * init.v; j < rs.domain.y;
         j += 2 * stride.v) // Only need even y positions.
      if (threadpos == 0)
        slice::ZSrc<T>::WriteGlobal(src, dst, srcpos, XY(i, j), rs.domain);
}

// Convert between external and global buffer for the y-plane source.
template <typename T, typename T1>
__dh__ void ConvertYSrc(T1 *src, T *dst, RunShape rs, int zshift, int threadpos,
                        UV warppos, UV blockpos) {
  UV init = warppos + rs.block * blockpos;
  UV stride = rs.block * rs.grid;
  for (int i = init.u; i < rs.domain.x; i += stride.u)
    if (init.v == 0)
      slice::YSrc<T>::WriteGlobal(src, dst, i, threadpos, rs.domain.x, rs.pml.n,
                                  zshift, defs::IsAux(threadpos, rs.pml.n));
}

} // namespace slice

#endif // _SLICE_H_
