// Coefficients defining the fdtd update along the z-axis.

#ifndef _ZCOEFF_H_
#define _ZCOEFF_H_

#include "defs.h"
#include "diamond.h"

namespace zcoeff {

using diamond::E;
using diamond::EffNz;
using diamond::Ehc;
using diamond::ExtZIndex;
using diamond::ExtZz;
using diamond::H;
using diamond::Nz;

// `Z` denotes the length of the unit cell along the z-axis, and `A` and `B`
// denote the update of the auxiliary PML fields. Each of these is defined for
// both E- and H-fields at every `z = z0`.
enum struct ExternalType { A, B, Z };
static constexpr const int NumExternalCoeffs = 6;

__dhce__ int ExternalComponentIndex(Ehc ehc, ExternalType type) {
  if (ehc == E) {
    if (type == ExternalType::A)
      return 0;
    else if (type == ExternalType::B)
      return 1;
    else // type == ExternalType::Z.
      return 2;
  } else { // ehc == H.
    if (type == ExternalType::A)
      return 3;
    else if (type == ExternalType::B)
      return 4;
    else // type == ExternalType::Z.
      return 5;
  }
}

// To avoid branching in the update code, we selectively map `ExternalType` to
// `InternalType` coefficients based on whether the thread is auxiliary or not.
//
// For threads which are not auxiliary, and do not have a corresponding
// auxiliary:
//   - `A` denotes the length of the cell in the z-direction, and
//   - `B` coefficient must be set to `0`.
//
// For threads which are auxiliary:
//   - `A` is ignored, and
//   - `B` is used as the scaling constant for the auxiliary fields.
//
// For threads which have a corresponding auxiliary thread:
//   - `A` must take into account both the length of the unit cell in the
//     z-direction as well as the scaling coefficient for the spatial derivative
//     in the auxiliary update.
//   - `B` corresponds to the scaling coefficient for the corresponding
//   auxiliary field.
//
enum struct InternalType { A, B };
static constexpr const int NumInternalCoeffs = 4;

__dhce__ int InternalComponentIndex(Ehc ehc, InternalType type) {
  if (ehc == E) {
    if (type == InternalType::A)
      return 0;
    else // type == InternalType::B.
      return 1;
  } else { // ehc == H.
    if (type == InternalType::A)
      return 2;
    else // type == InternalType::B.
      return 3;
  }
}

template <typename T> __dhce__ int ExternalElems(int npml) {
  return ExtZz<T>(npml) * NumExternalCoeffs;
}

__dhce__ int ExternalIndex(int z, Ehc ehc, ExternalType type) {
  return ExternalComponentIndex(ehc, type) + NumExternalCoeffs * z;
}

// Holds the coefficients for the thread.
template <typename T> struct ZCoeff {
  __dhce__ T Get(int k, Ehc ehc, InternalType type) {
    return values_[k][InternalComponentIndex(ehc, type)];
  }
  __dhce__ void Set(T value, int k, Ehc ehc, InternalType type) {
    values_[k][InternalComponentIndex(ehc, type)] = value;
  }

  T values_[Nz][NumInternalCoeffs];
};

// Loads a component from external to internal representation.
template <typename T, typename T1>
__dh__ void LoadComponent(ZCoeff<T> &zcoeff, int k, Ehc ehc, InternalType type,
                          T1 *ptr, int threadpos, int npml, int zshift,
                          bool isaux) {
  int z = ExtZIndex<T>(k, threadpos, npml, zshift);
  if (type == InternalType::A)
    zcoeff.Set(isaux ? defs::Zero<T>()
                     : ptr[ExternalIndex(z, ehc, ExternalType::A)] +
                           ptr[ExternalIndex(z, ehc, ExternalType::Z)],
               k, ehc, type);
  else // type == InternalType::B.
    zcoeff.Set(isaux ? ptr[ExternalIndex(z, ehc, ExternalType::B)]
                     : ptr[ExternalIndex(z, ehc, ExternalType::A)],
               k, ehc, type);
}

#ifndef __OMIT_HALF2__
// Specialized function for the `half2` case.
template <>
__dh__ void LoadComponent(ZCoeff<half2> &zcoeff, int k, Ehc ehc,
                          InternalType type, float *ptr, int threadpos,
                          int npml, int zshift, bool isaux) {
  int zlo = ExtZIndex<half2>(k, threadpos, npml, zshift);
  int zhi = ExtZIndex<half2>(k + Nz, threadpos, npml, zshift);
  if (type == InternalType::A)
    zcoeff.Set(isaux ? defs::Zero<half2>()
                     : __floats2half2_rn(
                           ptr[ExternalIndex(zlo, ehc, ExternalType::A)] +
                               ptr[ExternalIndex(zlo, ehc, ExternalType::Z)],
                           ptr[ExternalIndex(zhi, ehc, ExternalType::A)] +
                               ptr[ExternalIndex(zhi, ehc, ExternalType::Z)]),
               k, ehc, type);
  else // type == InternalType::B.
    zcoeff.Set(
        isaux
            ? __floats2half2_rn(ptr[ExternalIndex(zlo, ehc, ExternalType::B)],
                                ptr[ExternalIndex(zhi, ehc, ExternalType::B)])
            : __floats2half2_rn(ptr[ExternalIndex(zlo, ehc, ExternalType::A)],
                                ptr[ExternalIndex(zhi, ehc, ExternalType::A)]),
        k, ehc, type);
}
#endif

// Loads the coefficients into `zcoeff`.
template <typename T, typename T1>
__dh__ void Load(ZCoeff<T> &zcoeff, T1 *ptr, int threadpos, int npml,
                 int zshift, bool isaux) {
#pragma unroll
  for (int k = 0; k < Nz; ++k)
#pragma unroll
    for (Ehc ehc : {E, H})
#pragma unroll
      for (InternalType type : {InternalType::A, InternalType::B})
        LoadComponent(zcoeff, k, ehc, type, ptr, threadpos, npml, zshift,
                      isaux);
}

} // namespace zcoeff

#endif // _ZCOEFF_H_
