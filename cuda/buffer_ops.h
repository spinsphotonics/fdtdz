#ifndef _BUFFER_OPS_H_
#define _BUFFER_OPS_H_

#include "defs.h"
#include "diamond.h"

namespace buffer {
namespace {

using defs::kWarpSize;
using diamond::Cell;
using diamond::Node;
using diamond::X;
using diamond::Y;
using diamond::Z;

enum Type { UE, UH, VE, VH };

struct Node2 {
  __dhce__ Node2(Node n0, Node n1) : n0(n0), n1(n1) {}
  const Node n0, n1;
};

struct Node4 {
  __dhce__ Node4(Node n0, Node n1, Node n2, Node n3)
      : n0(n0), n1(n1), n2(n2), n3(n3) {}
  const Node n0, n1, n2, n3;
};

// Loads the 2 `nodes` from `val` to `cell`.
template <typename T>
__dh__ void Load(Cell<T> &cell, const Node2 node2, uint2 val) {
  cell.Set(val.x, node2.n0);
  cell.Set(val.y, node2.n1);
}

// Loads the 4 `nodes` from `val` to `cell`.
template <typename T>
__dh__ void Load(Cell<T> &cell, const Node4 node4, uint4 val) {
  cell.Set(val.x, node4.n0);
  cell.Set(val.y, node4.n1);
  cell.Set(val.z, node4.n2);
  cell.Set(val.w, node4.n3);
}

// Store 2 `nodes` of `cell` to a vectorized `uint2` at `ptr`,
template <typename T>
__dh__ void Store(Cell<T> &cell, const Node2 node2, uint2 *ptr) {
  ptr[0].x = cell.template Get<uint>(node2.n0);
  ptr[0].y = cell.template Get<uint>(node2.n1);
}

// Store 4 `nodes` of `cell` to a vectorized `uint4` at `ptr`,
template <typename T>
__dh__ void Store(Cell<T> &cell, const Node4 node4, uint4 *ptr) {
  ptr[0].x = cell.template Get<uint>(node4.n0);
  ptr[0].y = cell.template Get<uint>(node4.n1);
  ptr[0].z = cell.template Get<uint>(node4.n2);
  ptr[0].w = cell.template Get<uint>(node4.n3);
}

template <typename T>
__dh__ void Load(Cell<T> &cell, T *ptr, int threadpos, bool withfirst,
                 bool withlast, Node4 node40, Node4 node41, Node4 node42,
                 Node4 node43) {
  uint4 *ptr4 = ((uint4 *)ptr) + threadpos;
  if (withfirst)
    Load(cell, node40, ptr4[0 * kWarpSize]);
  Load(cell, node41, ptr4[1 * kWarpSize]);
  Load(cell, node42, ptr4[2 * kWarpSize]);
  if (withlast)
    Load(cell, node43, ptr4[3 * kWarpSize]);
}

template <typename T>
__dh__ void Load(Cell<T> &cell, T *ptr, int threadpos, bool withfirst,
                 bool withlast, Node2 node20, Node2 node21, Node2 node22,
                 Node4 node40, Node4 node41) {
  uint4 *ptr4 = ((uint4 *)ptr) + threadpos;
  Load(cell, node40, ptr4[(kWarpSize / 2) * (3 * 0 + 1)]);
  Load(cell, node41, ptr4[(kWarpSize / 2) * (3 * 1 + 1)]);

  uint2 *ptr2 = ((uint2 *)ptr) + threadpos;
  if (withfirst)
    Load(cell, node20, ptr2[kWarpSize * 3 * 0]);
  Load(cell, node21, ptr2[kWarpSize * 3 * 1]);
  if (withlast)
    Load(cell, node22, ptr2[kWarpSize * 3 * 2]);
}

template <typename T>
__dh__ void Store(Cell<T> &cell, T *ptr, int threadpos, bool withfirst,
                  bool withlast, Node4 node40, Node4 node41, Node4 node42,
                  Node4 node43) {
  uint4 *ptr4 = ((uint4 *)ptr) + threadpos;
  if (withfirst)
    Store(cell, node40, ptr4 + 0 * kWarpSize);
  Store(cell, node41, ptr4 + 1 * kWarpSize);
  Store(cell, node42, ptr4 + 2 * kWarpSize);
  if (withlast)
    Store(cell, node43, ptr4 + 3 * kWarpSize);
}

template <typename T>
__dh__ void Store(Cell<T> &cell, T *ptr, int threadpos, bool withfirst,
                  bool withlast, Node2 node20, Node2 node21, Node2 node22,
                  Node4 node40, Node4 node41) {
  uint4 *ptr4 = ((uint4 *)ptr) + threadpos;
  Store(cell, node40, ptr4 + (kWarpSize / 2) * (3 * 0 + 1));
  Store(cell, node41, ptr4 + (kWarpSize / 2) * (3 * 1 + 1));

  uint2 *ptr2 = ((uint2 *)ptr) + threadpos;
  if (withfirst)
    Store(cell, node20, ptr2 + kWarpSize * 3 * 0);
  Store(cell, node21, ptr2 + kWarpSize * 3 * 1);
  if (withlast)
    Store(cell, node22, ptr2 + kWarpSize * 3 * 2);
}

__dhce__ Node2 E2(int i, int j) {
  return Node2(Node(i, j, 0, diamond::E, Y), Node(i, j, 1, diamond::E, Y));
}

__dhce__ Node4 E4(int i, int j) {
  return Node4(Node(i, j, 0, diamond::E, X), //
               Node(i, j, 1, diamond::E, X), //
               Node(i + 1, j, 0, diamond::E, Z),
               Node(i + 1, j, 1, diamond::E, Z));
}

__dhce__ Node4 H4(int i0, int j0, diamond::Xyz xyz0, //
                  int i1, int j1, diamond::Xyz xyz1) {
  return Node4(
      Node(i0, j0, 0, diamond::H, xyz0), Node(i0, j0, 1, diamond::H, xyz0),
      Node(i1, j1, 0, diamond::H, xyz1), Node(i1, j1, 1, diamond::H, xyz1));
}

template <typename T>
__dh__ void Load(Cell<T> &cell, T *ptr, int threadpos, Type type,
                 bool includebot, bool includetop) {
  if (type == UE)
    Load(cell, ptr, threadpos, includebot, includetop, //
         E2(2, 0), E2(1, 1), E2(0, 2), E4(1, 1), E4(0, 2));
  else if (type == UH)
    Load(cell, ptr, threadpos, includebot, includetop, //
         H4(3, 0, X, 2, 0, Z), H4(2, 1, Y, 2, 1, X),   //
         H4(1, 1, Z, 1, 2, Y), H4(1, 2, X, 0, 2, Z));
  else if (type == VE)
    Load(cell, ptr, threadpos, includebot, includetop, //
         E2(0, -2), E2(1, -1), E2(2, 0), E4(0, -1), E4(1, 0));
  else                                                   // type == VH.
    Load(cell, ptr, threadpos, includebot, includetop,   //
         H4(1, -2, X, 0, -2, Z), H4(1, -1, Y, 2, -1, X), //
         H4(1, -1, Z, 2, 0, Y), H4(3, 0, X, 2, 0, Z));
}

template <typename T>
__dh__ void LoadBottomPointV(Cell<T> &cell, T *ptr, int threadpos) {
  uint4 *ptr4 = ((uint4 *)ptr) + threadpos;
  Load(cell, H4(1, 2, X, 0, 2, Z), ptr4[0]);
}

template <typename T>
__dh__ void Store(Cell<T> &cell, T *ptr, int threadpos, Type type,
                  bool includebot, bool includetop) {
  if (type == UE)
    Store(cell, ptr, threadpos, includebot, includetop, //
          E2(0, -2), E2(-1, -1), E2(-2, 0), E4(-1, -1), E4(-2, 0));
  else if (type == UH)
    Store(cell, ptr, threadpos, includebot, includetop,   //
          H4(1, -2, X, 0, -2, Z), H4(0, -1, Y, 0, -1, X), //
          H4(-1, -1, Z, -1, 0, Y), H4(-1, 0, X, -2, 0, Z));
  else if (type == VE)
    Store(cell, ptr, threadpos, includebot, includetop, //
          E2(-2, 0), E2(-1, 1), E2(0, 2), E4(-2, 1), E4(-1, 2));
  else                                                   // type == VH.
    Store(cell, ptr, threadpos, includebot, includetop,  //
          H4(-1, 0, X, -2, 0, Z), H4(-1, 1, Y, 0, 1, X), //
          H4(-1, 1, Z, 0, 2, Y), H4(1, 2, X, 0, 2, Z));
}

} // namespace
} // namespace buffer

#endif // _BUFFER_OPS_H_
