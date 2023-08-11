// Allows for writing to output fields.

#ifndef _FIELD_H_
#define _FIELD_H_

#include "defs.h"
#include "diamond.h"

namespace field {

using defs::RunShape;
using defs::XY;
using diamond::Cell;
using diamond::ExtZIndex;
using diamond::ExtZz;
using diamond::N;
using diamond::Node;

template <typename T>
__dhce__ int ExternalElems(RunShape::Vol sub, int nout, int npml) {
  int xx = sub.x1 - sub.x0 + 2 * N;
  int yy = sub.y1 - sub.y0 + 2 * N;
  int zz = sub.z1 - sub.z0;
  return xx * yy * zz * diamond::kNumXyz * nout;
}

__dhce__ int ExtNodeIndex(Node n, int outindex, RunShape::Vol sub) {
  int xx = sub.x1 - sub.x0 + 2 * N;
  int yy = sub.y1 - sub.y0 + 2 * N;
  int zz = sub.z1 - sub.z0;
  return n.k + zz * (n.j + yy * (n.i + xx * (diamond::Index(n.xyz) +
                                             diamond::kNumXyz * outindex)));
}

template <typename T>
__dhce__ int IntNodeIndex(Node n, int outindex, int threadpos, XY pos,
                          XY domain, int npml, int zshift, RunShape::Vol sub) {
  Node externalnode(n.i + pos.x - sub.x0 + N,                            //
                    n.j + pos.y - sub.y0 + N,                            //
                    ExtZIndex<T>(n.k, threadpos, npml, zshift) - sub.z0, //
                    n.ehc,                                               //
                    n.xyz);
  return ExtNodeIndex(externalnode, outindex, sub);
}

// Write a specific node to the output buffer.
template <typename T, typename T1>
__dhce__ void WriteNode(const Cell<T> &cell, T1 *ptr, Node n, XY pos,
                        int outindex, int threadpos, XY domain, int npml,
                        int zshift, RunShape::Vol sub) {
  int extzindex = ExtZIndex<T>(n.k, threadpos, npml, zshift);
  if (extzindex >= sub.z0 && extzindex < sub.z1) {
    int index =
        IntNodeIndex<T>(n, outindex, threadpos, pos, domain, npml, zshift, sub);
    ptr[index] = cell.Get(n);
  }
}

#ifndef __OMIT_HALF2__
// Specialized for `half2` case.
template <>
__dh__ void WriteNode(const Cell<half2> &cell, float *ptr, Node n, XY pos,
                      int outindex, int threadpos, XY domain, int npml,
                      int zshift, RunShape::Vol sub) {
  {
    int extzindex = ExtZIndex<half2>(n.k, threadpos, npml, zshift);
    if (extzindex >= sub.z0 && extzindex < sub.z1)
      ptr[IntNodeIndex<half2>(n, outindex, threadpos, pos, domain, npml, zshift,
                              sub)] = __low2float(cell.Get(n));
  }

  {
    int extzindex =
        ExtZIndex<half2>(n.k + diamond::Nz, threadpos, npml, zshift);
    if (extzindex >= sub.z0 && extzindex < sub.z1)
      ptr[IntNodeIndex<half2>(n.dK(diamond::Nz), outindex, threadpos, pos,
                              domain, npml, zshift, sub)] =
          __high2float(cell.Get(n));
  }
}
#endif

// Write node values to the external output buffer.
template <typename T, typename T1>
__dhce__ void WriteCell(const Cell<T> &cell, T1 *ptr, XY pos, int outindex,
                        int threadpos, XY domain, int npml, int zshift,
                        bool isaux, RunShape::Vol sub) {
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
                      zshift, sub);
          }
        }
}

template <typename T, typename T1>
__dh__ void Init(T1 *ptr, RunShape rs, int threadpos, defs::UV warppos,
                 defs::UV blockpos) {
  int init =
      threadpos +
      defs::kWarpSize *
          (warppos.u +
           rs.block.u * (warppos.v +
                         rs.block.v * (blockpos.u + rs.grid.u * blockpos.v)));
  int stride = defs::kWarpSize * defs::Prod(rs.block * rs.grid);
  for (int i = init; i < ExternalElems<T>(rs.sub, rs.out.num, rs.pml.n);
       i += stride)
    ptr[i] = defs::Zero<T1>();
}

} // namespace field

#endif // _FIELD_H_
