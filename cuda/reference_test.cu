// Test the reference simulation.

#include <gtest/gtest.h>

#include "defs.h"
#include "diamond.h"
#include "reference.h"
#include "testutils.h"

namespace reference {
namespace {

using defs::RunShape;
using defs::UV;
using defs::XY;
using diamond::Node;

// Not actual energy, since it ignores material, time-offset, etc...
// That said, it should be good enough to measure that a pml layer is actually
// doing something.
template <typename T>
T EnergyAtStep(int step, SimParams<T> sp, Cache<T> &cache) {
  T val = Zero<T>();
  for (Ehc ehc : {E, H})
    for (Xyz xyz : {X, Y, Z})
      for (int i = 0; i < sp.x; ++i)
        for (int j = 0; j < sp.y; ++j)
          for (int k = 0; k < sp.z; ++k) {
            T f = Get(Node(i, j, k, ehc, xyz), step, FIELD, sp, cache);
            val += f * f;
          }
  return val;
}

// Without an existing cache.
template <typename T> T EnergyAtStep(int step, SimParams<T> sp) {
  Cache<T> cache;
  return EnergyAtStep(step, sp, cache);
}

template <typename T> void PrintEnergy(int numsteps, SimParams<T> sp) {
  Cache<T> cache;
  for (int step = 0; step < numsteps; ++step) {
    std::cout << "step: " << step
              << ", energy: " << EnergyAtStep(step, sp, cache) << "\n";
  }
}

TEST(Reference, BasicStencil) {
  int xx = 9;
  int yy = 9;
  int zz = 9;
  int tt = 1;
  int nelems = FieldElems(xx, yy, zz);
  Node n(2, 5, 6, E, Z);

  SimAlloc<int> alloc(xx, yy, zz, tt, /*dt=*/1, /*srcnode=*/n);
  SimParams<int> sp = alloc.Params();

  // sp.src0[FieldIndex(n, sp.x, sp.y)] = 1;
  sp.wf0[0] = 1;

  Cache<int> cache;
  EXPECT_EQ(Get(n, 0, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n.AsHx(), 0, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsHx().dJ(-1), 0, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsHy(), 0, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsHy().dI(-1), 0, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n.AsEx(), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsEx().dK(+1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEx().dI(-1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEx().dI(-1).dK(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsEy(), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsEy().dK(+1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEy().dJ(-1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEy().dJ(-1).dK(+1), 1, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n, 1, FIELD, sp, cache), -3);
  EXPECT_EQ(Get(n.dI(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dI(-1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dJ(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dJ(-1), 1, FIELD, sp, cache), 1);
}

TEST(Reference, BasicStencilWithModifications) {
  int xx = 9;
  int yy = 9;
  int zz = 9;
  int tt = 1;
  int nelems = FieldElems(xx, yy, zz);
  Node n(2, 5, 6, E, Z);

  SimAlloc<int> alloc(xx, yy, zz, tt, /*dt=*/1, /*srcnode=*/n);
  SimParams<int> sp = alloc.Params();

  // Use the other source
  sp.wf1[0] = 1;

  // Modify `mat` and `abs`.
  sp.mat[FieldIndex(n.AsEx(), sp.x, sp.y, sp.z)] = 0;
  sp.abs[FieldIndex(n, sp.x, sp.y)] = -4;
  sp.zcoeff[n.k].edz = 2;

  Cache<int> cache;
  EXPECT_EQ(Get(n, 0, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n.AsHx(), 0, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsHx().dJ(-1), 0, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsHy(), 0, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsHy().dI(-1), 0, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n.AsEx(), 1, FIELD, sp, cache), 0);
  EXPECT_EQ(Get(n.AsEx().dK(+1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEx().dI(-1), 1, FIELD, sp, cache), -2);
  EXPECT_EQ(Get(n.AsEx().dI(-1).dK(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsEy(), 1, FIELD, sp, cache), 2);
  EXPECT_EQ(Get(n.AsEy().dK(+1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEy().dJ(-1), 1, FIELD, sp, cache), -2);
  EXPECT_EQ(Get(n.AsEy().dJ(-1).dK(+1), 1, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n, 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dI(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dI(-1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dJ(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dJ(-1), 1, FIELD, sp, cache), 1);
}

TEST(Reference, BasicStencilWithPml) {
  int xx = 9;
  int yy = 9;
  int zz = 9;
  int tt = 1;
  int nelems = FieldElems(xx, yy, zz);
  Node n(2, 5, 6, E, Z);

  SimAlloc<int> alloc(xx, yy, zz, tt, /*dt=*/1, /*srcnode=*/n);
  SimParams<int> sp = alloc.Params();

  // Use the other source
  sp.wf1[0] = 1;

  // Modify pml coefficients.
  sp.zcoeff[n.k + 1].epa = -1;
  sp.zcoeff[n.k + 1].epb = 0;

  Cache<int> cache;
  EXPECT_EQ(Get(n, 0, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n.AsHx(), 0, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsHx().dJ(-1), 0, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsHy(), 0, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsHy().dI(-1), 0, FIELD, sp, cache), 1);

  EXPECT_EQ(Get(n.AsEx(), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsEx().dK(+1), 1, FIELD, sp, cache), 0);
  EXPECT_EQ(Get(n.AsEx().dI(-1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEx().dI(-1).dK(+1), 1, FIELD, sp, cache), 0);
  EXPECT_EQ(Get(n.AsEy(), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.AsEy().dK(+1), 1, FIELD, sp, cache), 0);
  EXPECT_EQ(Get(n.AsEy().dJ(-1), 1, FIELD, sp, cache), -1);
  EXPECT_EQ(Get(n.AsEy().dJ(-1).dK(+1), 1, FIELD, sp, cache), 0);

  EXPECT_EQ(Get(n, 1, FIELD, sp, cache), -3);
  EXPECT_EQ(Get(n.dI(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dI(-1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dJ(+1), 1, FIELD, sp, cache), 1);
  EXPECT_EQ(Get(n.dJ(-1), 1, FIELD, sp, cache), 1);
}

TEST(Reference, PmlAbsorption) {
  int xx = 9;
  int yy = 9;
  int zz = 9;
  int tt = 1;
  int nelems = FieldElems(xx, yy, zz);
  Node n(4, 4, 4, E, Y);

  // NOTE: `0.577` satisfies Courant condition.
  SimAlloc<float> alloc(xx, yy, zz, tt, /*dt=*/0.577f, /*srcnode=*/n);
  SimParams<float> sp = alloc.Params();

  Fill(0.0f, sp.abs, nelems);
  Fill(0.577f, sp.mat, nelems);
  Fill(ZCoeff<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, sp.zcoeff, zz);

  sp.wf0[0] = 1;

  EXPECT_GT(EnergyAtStep(/*step=*/40, sp), EnergyAtStep(/*step=*/0, sp) / 2.0f)
      << "Energy without PML must be greater than 50% of energy at step 0";

  // Modify pml coefficients.
  for (int i = 0; i < 3; ++i) {
    sp.zcoeff[i] = ZCoeff<float>{1.0f, -0.3f, 0.7f, 1.0f, -0.3f, 0.7f};
    sp.zcoeff[zz - i - 1] = ZCoeff<float>{1.0f, -0.3f, 0.7f, 1.0f, -0.3f, 0.7f};
  }

  EXPECT_LE(EnergyAtStep(/*step=*/40, sp), EnergyAtStep(/*step=*/0, sp) / 2.0f)
      << "Energy withPML must be less than 50% of energy at step 0";
}

} // namespace
} // namespace reference
