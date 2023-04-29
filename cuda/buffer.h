// Functions for accessing block- and grid-level buffers.

#ifndef _BUFFER_H_
#define _BUFFER_H_

#include "buffer_indexing.h"
#include "buffer_ops.h"
#include "defs.h"
#include "diamond.h"
#include "scanner.h"

namespace buffer {

using defs::kWarpSize;
using defs::UV;
using diamond::Cell;
using diamond::E;
using diamond::H;
using diamond::N;
using diamond::Node;
using diamond::Nz;

__dhce__ int SharedElems(RunShape rs) {
  return defs::Sum(SharedBlockIndexing::NumElems(rs.block));
}

__dhce__ UV GlobalElemsUV(RunShape rs) {
  return scanner::BufferShape(rs) * defs::VU(rs.grid) *
         GlobalBlockIndexing::NumElems(rs.block);
}

__dhce__ UV GlobalIndex(UV warppos, UV blockpos, UV bufpos, RunShape rs,
                        Ehc ehc) {
  return
      // Place u-buffer before v-buffer.
      UV(0, GlobalElemsUV(rs).u) +

      // Navigate to the correct `(warp, block, buffer)` position.
      GlobalBlockIndexing::Index(warppos, rs.block, ehc) +
      GlobalBlockIndexing::NumElems(rs.block) *
          (defs::VU(blockpos) + defs::VU(rs.grid) * bufpos);
}

__dhce__ int GlobalElems(RunShape rs) { return defs::Sum(GlobalElemsUV(rs)); }

template <typename T>
__dh__ void LoadSharedTopPointH(Cell<T> &cell, T *ptr, int threadpos,
                                UV warppos, UV blockshape) {
  // Load the top leading point of the current diamond from the bottom leading
  // point of the diamond above it.
  int offset = SharedBlockIndexing::Index(warppos - UV(1, 0), blockshape, H).v;
  LoadBottomPointV(cell, ptr + offset, threadpos);
}

template <typename T>
__dh__ void LoadSharedU(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                        UV blockshape, Ehc ehc) {
  if (ehc == H && !defs::IsTrailV(warppos, blockshape)) {
    LoadSharedTopPointH(cell, ptr, threadpos, warppos, blockshape);
  }
  int offset =
      SharedBlockIndexing::Index(warppos - UV(1, 0), blockshape, ehc).u;
  Load(cell, ptr + offset, threadpos, (ehc == H ? UH : UE),
       /*includebot=*/false,
       /*includetop=*/ehc == H && defs::IsTrailV(warppos, blockshape));
}

template <typename T>
__dh__ void LoadSharedV(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                        UV blockshape, Ehc ehc) {
  int offset =
      SharedBlockIndexing::Index(warppos - UV(0, 1), blockshape, ehc).v;
  Load(cell, ptr + offset, threadpos, (ehc == H ? VH : VE),
       /*includebot=*/ehc == H,
       /*includetop=*/!defs::IsLeadU(warppos));
}

template <typename T>
__dh__ void LoadGlobalU(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                        UV blockpos, UV bufpos, RunShape rs, Ehc ehc) {
  int offset = GlobalIndex(warppos, blockpos, bufpos, rs, ehc).u;
  Load(cell, ptr + offset, threadpos, (ehc == H ? UH : UE),
       /*includebot=*/!defs::IsLeadV(warppos),
       /*includetop=*/ehc == H || IsTrailV(warppos, rs.block));
}

template <typename T>
__dh__ void LoadGlobalV(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                        UV blockpos, UV bufpos, RunShape rs, Ehc ehc) {
  int offset = GlobalIndex(warppos, blockpos, bufpos, rs, ehc).v;
  Load(cell, ptr + offset, threadpos, (ehc == H ? VH : VE),
       /*includebot=*/ehc == H,
       /*includetop=*/true);
}

template <typename T>
__dh__ void StoreSharedU(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                         UV blockshape, Ehc ehc) {
  int offset = SharedBlockIndexing::Index(warppos, blockshape, ehc).u;
  Store(cell, ptr + offset, threadpos, (ehc == H ? UH : UE),
        /*includebot=*/false,
        /*includetop=*/ehc == H && defs::IsTrailV(warppos, blockshape));
}

template <typename T>
__dh__ void StoreSharedV(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                         UV blockshape, Ehc ehc) {
  int offset = SharedBlockIndexing::Index(warppos, blockshape, ehc).v;
  Store(cell, ptr + offset, threadpos, (ehc == H ? VH : VE),
        /*includebot=*/ehc == H || !defs::IsTrailU(warppos, blockshape),
        /*includetop=*/false);
}

template <typename T>
__dh__ void StoreGlobalU(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                         UV blockpos, UV bufpos, RunShape rs, Ehc ehc) {
  int offset = GlobalIndex(warppos, blockpos, bufpos, rs, ehc).u;
  Store(cell, ptr + offset, threadpos, (ehc == H ? UH : UE),
        /*includebot=*/false,
        /*includetop=*/true);
}

template <typename T>
__dh__ void StoreGlobalV(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                         UV blockpos, UV bufpos, RunShape rs, Ehc ehc) {
  int offset = GlobalIndex(warppos, blockpos, bufpos, rs, ehc).v;
  Store(cell, ptr + offset, threadpos, (ehc == H ? VH : VE),
        /*includebot=*/ehc == H || !defs::IsTrailU(warppos, rs.block),
        /*includetop=*/defs::IsLeadU(warppos));
}

template <typename T>
__dh__ void LoadShared(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                       UV blockshape, Ehc ehc) {
  if (!defs::IsLeadU(warppos))
    LoadSharedU(cell, ptr, threadpos, warppos, blockshape, ehc);
  if (!defs::IsLeadV(warppos))
    LoadSharedV(cell, ptr, threadpos, warppos, blockshape, ehc);
}

template <typename T>
__dh__ void LoadGlobal(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                       UV blockpos, UV bufpos, RunShape rs, Ehc ehc) {
  if (defs::IsLeadU(warppos))
    LoadGlobalU(cell, ptr, threadpos, warppos, blockpos, bufpos, rs, ehc);
  if (defs::IsLeadV(warppos))
    LoadGlobalV(cell, ptr, threadpos, warppos, blockpos, bufpos, rs, ehc);
}

template <typename T>
__dh__ void StoreShared(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                        UV blockshape, Ehc ehc) {
  if (!defs::IsTrailU(warppos, blockshape))
    StoreSharedU(cell, ptr, threadpos, warppos, blockshape, ehc);
  if (!defs::IsTrailV(warppos, blockshape))
    StoreSharedV(cell, ptr, threadpos, warppos, blockshape, ehc);
}

template <typename T>
__dh__ void StoreGlobal(Cell<T> &cell, T *ptr, int threadpos, UV warppos,
                        UV blockpos, UV bufpos, RunShape rs, Ehc ehc) {
  if (defs::IsTrailU(warppos, rs.block))
    StoreGlobalU(cell, ptr, threadpos, warppos, blockpos, bufpos, rs, ehc);
  if (defs::IsTrailV(warppos, rs.block))
    StoreGlobalV(cell, ptr, threadpos, warppos, blockpos, bufpos, rs, ehc);
}

template <typename T>
__dh__ void Init(T *ptr, RunShape rs, int threadpos, UV warppos, UV blockpos) {
  int init =
      threadpos +
      kWarpSize *
          (warppos.u +
           rs.block.u * (warppos.v +
                         rs.block.v * (blockpos.u + rs.grid.u * blockpos.v)));
  int stride = kWarpSize * defs::Prod(rs.block * rs.grid);
  for (int i = init; i < GlobalElems(rs); i += stride)
    ptr[i] = defs::Zero<T>();
}

} // namespace buffer

#endif // _BUFFER_H_
