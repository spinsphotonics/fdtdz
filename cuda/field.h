// Allows for writing to output fields.

#ifndef _FIELD_H_
#define _FIELD_H_

#include "defs.h"
#include "diamond.h"

namespace field {

using defs::XY;
using diamond::Cell;
using diamond::ExtZIndex;
using diamond::ExtZz;
using diamond::Node;

// template <typename T>
// __dhce__ int ExternalElems(XY domain, int nout, int npml) {
//   return defs::Prod(domain) * ExtZz<T>(npml) * diamond::kNumXyz * nout;
// }
//
// __dhce__ int ExtNodeIndex(Node n, int outindex, int xx, int yy, int zz) {
//   return n.k + zz * (n.j + yy * (n.i + xx * (diamond::Index(n.xyz) +
//                                              diamond::kNumXyz * outindex)));
// }
//
// template <typename T>
// __dhce__ int IntNodeIndex(Node n, int outindex, int threadpos, XY pos,
//                           XY domain, int npml, int zshift) {
//   Node externalnode(n.i + pos.x,                                //
//                     n.j + pos.y,                                //
//                     ExtZIndex<T>(n.k, threadpos, npml, zshift), //
//                     n.ehc,                                      //
//                     n.xyz);
//   return ExtNodeIndex(externalnode, outindex, domain.x, domain.y,
//                       ExtZz<T>(npml));
// }
//
// // Write a specific node to the output buffer.
// template <typename T, typename T1>
// __dhce__ void WriteNode(const Cell<T> &cell, T1 *ptr, Node n, XY pos,
//                         int outindex, int threadpos, XY domain, int npml,
//                         int zshift) {
//   int index =
//       IntNodeIndex<T>(n, outindex, threadpos, pos, domain, npml, zshift);
//   ptr[index] = cell.Get(n);
// }
//
// #ifndef __OMIT_HALF2__
// // Specialized for `half2` case.
// template <>
// __dh__ void WriteNode(const Cell<half2> &cell, float *ptr, Node n, XY pos,
//                       int outindex, int threadpos, XY domain, int npml,
//                       int zshift) {
//   ptr[IntNodeIndex<half2>(n, outindex, threadpos, pos, domain, npml, zshift)]
//   =
//       __low2float(cell.Get(n));
//   ptr[IntNodeIndex<half2>(n.dK(diamond::Nz), outindex, threadpos, pos,
//   domain,
//                           npml, zshift)] = __high2float(cell.Get(n));
// }
// #endif
//
// // Write node values to the external output buffer.
// template <typename T, typename T1>
// __dhce__ void WriteCell(const Cell<T> &cell, T1 *ptr, XY pos, int outindex,
//                         int threadpos, XY domain, int npml, int zshift,
//                         bool isaux) {
// #pragma unroll
//   for (int i : diamond::AllI)
// #pragma unroll
//     for (int j : diamond::AllJ)
// #pragma unroll
//       for (diamond::Xyz xyz : diamond::AllXyz)
// #pragma unroll
//         for (int k : diamond::AllK) {
//           diamond::Node n(i, j, k, diamond::E, xyz);
//           if (diamond::IsInsideDiamond(n) && !isaux) {
//             WriteNode(cell, ptr, n, pos, outindex, threadpos, domain, npml,
//                       zshift);
//           }
//         }
// }

template <typename T>
__dhce__ int ExternalElems(defs::RunShape::Out::Range xrange,
                           defs::RunShape::Out::Range yrange,
                           defs::RunShape::Out::Range zrange, //
                           int nout, int npml) {
  return (xrange.stop - xrange.start) * //
         (yrange.stop - yrange.start) * //
         (zrange.stop - zrange.start) * //
         diamond::kNumXyz * nout;
}

__dhce__ int ExtNodeIndex(Node n, int outindex,
                          defs::RunShape::Out::Range xrange,
                          defs::RunShape::Out::Range yrange,
                          defs::RunShape::Out::Range zrange) {
  int xx = xrange.stop - xrange.start;
  int yy = yrange.stop - yrange.start;
  int zz = zrange.stop - zrange.start;
  return n.k + zz * (n.j + yy * (n.i + xx * (diamond::Index(n.xyz) +
                                             diamond::kNumXyz * outindex)));
}

template <typename T>
__dhce__ int IntNodeIndex(Node n, int outindex, int threadpos, XY pos,
                          XY domain, int npml, int zshift,
                          defs::RunShape::Out::Range xrange,
                          defs::RunShape::Out::Range yrange,
                          defs::RunShape::Out::Range zrange) {
  Node externalnode(n.i + pos.x - xrange.start, //
                    n.j + pos.y - yrange.start, //
                    ExtZIndex<T>(n.k, threadpos, npml, zshift) -
                        zrange.start, //
                    n.ehc,            //
                    n.xyz);
  return ExtNodeIndex(externalnode, outindex, xrange, yrange, zrange);
}

// Write a specific node to the output buffer.
template <typename T, typename T1>
__dhce__ void WriteNode(const Cell<T> &cell, T1 *ptr, Node n, XY pos,
                        int outindex, int threadpos, XY domain, int npml,
                        int zshift, defs::RunShape::Out::Range xrange,
                        defs::RunShape::Out::Range yrange,
                        defs::RunShape::Out::Range zrange) {
  int extzindex = ExtZIndex<T>(n.k, threadpos, npml, zshift);
  if (extzindex >= zrange.start && extzindex < zrange.stop) {
    int index = IntNodeIndex<T>(n, outindex, threadpos, pos, domain, npml,
                                zshift, xrange, yrange, zrange);
    ptr[index] = cell.Get(n);
  }
}

#ifndef __OMIT_HALF2__
// Specialized for `half2` case.
template <>
__dh__ void WriteNode(const Cell<half2> &cell, float *ptr, Node n, XY pos,
                      int outindex, int threadpos, XY domain, int npml,
                      int zshift, defs::RunShape::Out::Range xrange,
                      defs::RunShape::Out::Range yrange,
                      defs::RunShape::Out::Range zrange) {
  {
    int extzindex = ExtZIndex<half2>(n.k, threadpos, npml, zshift);
    if (extzindex >= zrange.start && extzindex < zrange.stop)
      ptr[IntNodeIndex<half2>(n, outindex, threadpos, pos, domain, npml, zshift,
                              xrange, yrange, zrange)] =
          __low2float(cell.Get(n));
  }

  {
    int extzindex =
        ExtZIndex<half2>(n.k + diamond::Nz, threadpos, npml, zshift);
    if (extzindex >= zrange.start && extzindex < zrange.stop)
      ptr[IntNodeIndex<half2>(n.dK(diamond::Nz), outindex, threadpos, pos,
                              domain, npml, zshift, xrange, yrange, zrange)] =
          __high2float(cell.Get(n));
  }
}
#endif

// Write node values to the external output buffer.
template <typename T, typename T1>
__dhce__ void WriteCell(const Cell<T> &cell, T1 *ptr, XY pos, int outindex,
                        int threadpos, XY domain, int npml, int zshift,
                        bool isaux, defs::RunShape::Out::Range xrange,
                        defs::RunShape::Out::Range yrange,
                        defs::RunShape::Out::Range zrange) {
#pragma unroll
  for (int i : diamond::AllI)
#pragma unroll
    for (int j : diamond::AllJ)
#pragma unroll
      for (diamond::Xyz xyz : diamond::AllXyz)
#pragma unroll
        for (int k : diamond::AllK) {
          diamond::Node n(i, j, k, diamond::E, xyz);
          if (diamond::IsInsideDiamond(n) && !isaux) {
            WriteNode(cell, ptr, n, pos, outindex, threadpos, domain, npml,
                      zshift, xrange, yrange, zrange);
          }
        }
}

template <typename T, typename T1>
__dh__ void Init(T1 *ptr, defs::RunShape rs, int threadpos, defs::UV warppos,
                 defs::UV blockpos) {
  int init =
      threadpos +
      defs::kWarpSize *
          (warppos.u +
           rs.block.u * (warppos.v +
                         rs.block.v * (blockpos.u + rs.grid.u * blockpos.v)));
  int stride = defs::kWarpSize * defs::Prod(rs.block * rs.grid);
  for (int i = init;
       i < ExternalElems<T>(rs.out.x, rs.out.y, rs.out.z, rs.out.num, rs.pml.n);
       i += stride)
    ptr[i] = defs::Zero<T1>();
}

} // namespace field

#endif // _FIELD_H_
