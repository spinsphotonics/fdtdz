#include <gtest/gtest.h>

#include "defs.h"
#include "diamond.h"
#include "reference.h"
#include "testutils.h"
#include "update.h"
#include "verification.h"

namespace verification {
namespace {

using defs::kWarpSize;
using defs::RunShape;
using defs::UV;
using defs::XY;
using diamond::E;
using diamond::Node;
using diamond::Nz;
using diamond::X;
using diamond::Xyz;
using diamond::Y;
using diamond::Z;
using reference::FieldIndex;

template <typename T>
void FillWithXyHalo(T val, T haloval, int d, T *ptr,
                    reference::SimParams<T> sp) {
  for (int i = 0; i < sp.x; ++i)
    for (int j = 0; j < sp.y; ++j)
      for (int k = 0; k < sp.z; ++k)
        for (Xyz xyz : diamond::AllXyz) {
          int fieldindex = FieldIndex(Node(i, j, k, E, xyz), sp.x, sp.y, sp.z);
          if (i >= d && i < sp.x - d && j >= d && j < sp.y - d)
            ptr[fieldindex] = val;
          else
            ptr[fieldindex] = haloval;
        }
}

template <typename T>
void PrintField(T *ptr, int k, Xyz xyz, RunShape::Out::Range xrange,
                RunShape::Out::Range yrange, RunShape::Out::Range zrange) {
  int xx = xrange.stop - xrange.start;
  int yy = yrange.stop - yrange.start;
  int zz = zrange.stop - zrange.start;
  k -= zrange.start;
  for (int j = yy - 1; j >= 0; --j) {
    for (int i = 0; i < xx; ++i)
      std::cout << ptr[FieldIndex(Node(i, j, k, E, xyz), xx, yy, zz)] << " ";
    std::cout << "\n";
  }
}

template <typename T>
void PrintField2(T *ptr, int i, Xyz xyz, reference::SimParams<T> sp) {
  for (int j = sp.y - 1; j >= 0; --j) {
    for (int k = 0; k < sp.z; ++k)
      std::cout << ptr[FieldIndex(Node(i, j, k, E, xyz), sp.x, sp.y, sp.z)]
                << " ";
    std::cout << "\n";
  }
}

template <typename T, typename T1, int Npml> struct PointSim {
  PointSim(T1 mat0, Node srcnode, int timestep, int nlo, int nhi,
           RunShape::Src::Type srctype = RunShape::Src::ZSLICE,
           RunShape::Out::Range xrange = RunShape::Out::Range(0, 16),
           RunShape::Out::Range yrange = RunShape::Out::Range(0, 16),
           RunShape::Out::Range zrange =
               RunShape::Out::Range(0, diamond::ExtZz<T>(Npml)))
      : rs(/*block=*/UV(2, 2),
           /*grid=*/UV(2, 2),
           /*spacing=*/2,
           /*domain=*/XY(16, 16),
           /*pml=*/RunShape::Pml(/*n=*/Npml, /*zshift=*/nhi),
           /*src=*/
           RunShape::Src(srctype,
                         // For YSLICE sources, round to next even number,
                         // and expect wf1 to be set instead. This is
                         // because of the way the ysrc is implemented...
                         /*srcpos=*/srctype == RunShape::Src::ZSLICE
                             ? srcnode.k + nhi
                             : srcnode.j + (srcnode.j % 2)),
           /*out=*/
           RunShape::Out(
               /*start=*/timestep, /*interval=*/1, /*num=*/1, xrange, yrange,
               zrange)),
        alloc(rs.domain.x, rs.domain.y, diamond::ExtZz<T>(Npml), timestep, mat0,
              srcnode),
        sp(alloc.Params()), srcnode_(srcnode), timestep_(timestep), nlo_(nlo),
        nhi_(nhi) {
    // Fill with halo.
    FillWithXyHalo(mat0, Zero<T1>(), diamond::N, sp.mat, sp);
  }

  reference::SimParams<T1> SimParams() { return sp; }

  void Sim(T1 *outptr) {
    ASSERT_TRUE(scanner::IsValidRunShape(rs));
    ASSERT_EQ(XY(sp.x, sp.y), rs.domain);

    // Now build and run the simulation kernel.
    RunKernel<T, T1, Npml>(rs, outptr, sp, nlo_, nhi_);
  }

  void SimAndTest() {
    int xx = rs.out.x.stop - rs.out.x.start;
    int yy = rs.out.y.stop - rs.out.y.start;
    int zz = rs.out.z.stop - rs.out.z.start;

    testutils::Array<T1> out(reference::FieldElems(xx, yy, zz));
    Sim(out.Ptr());

    std::cout << "\nReference:\n";
    reference::PrintZSlice(/*k=*/srcnode_.k, E, srcnode_.xyz, timestep_,
                           reference::FIELD, sp);

    std::cout << "\nKernel2:\n";
    PrintField(out.Ptr(), /*k=*/srcnode_.k, srcnode_.xyz, rs.out.x, rs.out.y,
               rs.out.z);
    // PrintField2(out.Ptr(), /*i=*/srcnode_.i, srcnode_.xyz, sp);

    // Check.
    reference::Cache<T1> cache;
    for (int x = diamond::N; x < xx - diamond::N; ++x)
      for (int y = diamond::N; y < yy - diamond::N; ++y)
        for (int z = 0; z < zz; ++z)
          for (Xyz xyz : diamond::AllXyz) {
            Node refnode(x + rs.out.x.start, //
                         y + rs.out.y.start, //
                         z + rs.out.z.start, //
                         E, xyz);
            Node outnode(x, y, z, E, xyz);
            ASSERT_FLOAT_EQ(
                reference::Get(refnode, timestep_, reference::FIELD, sp, cache),
                out.Ptr()[FieldIndex(outnode, xx, yy, zz)])
                << "refnode: " << refnode << ", outnode: " << outnode;
          }
  }

private:
  RunShape rs;
  reference::SimAlloc<T1> alloc;
  reference::SimParams<T1> sp;
  const Node srcnode_;
  const int timestep_, nlo_, nhi_;
};

TEST(Verification, PointSource) {
  Node srcnode(7, 7, 17, E, X);
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/1.0f, srcnode,
                                         /*timestep=*/8,
                                         /*nlo=*/0, /*nhi=*/0);
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.0f;
  sim.SimAndTest();
}

TEST(Verification, PointSourceSubvolume) {
  Node srcnode(7, 7, 17, E, X);
  PointSim<float, float, /*Npml=*/0> sim(
      /*mat0=*/1.0f, srcnode,
      /*timestep=*/8,
      /*nlo=*/0,
      /*nhi=*/0,
      /*srctype=*/RunShape::Src::ZSLICE,
      /*xrange=*/RunShape::Out::Range(2, 14),
      /*yrange=*/RunShape::Out::Range(2, 14),
      /*zrange=*/
      RunShape::Out::Range(5, diamond::ExtZz<float>(/*Npml=*/0) - 5));
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.0f;
  sim.SimAndTest();
}

TEST(Verification, PointSourceWithModifications) {
  Node srcnode(7, 7, 16, E, Y);
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/1.0f, srcnode,
                                         /*timestep=*/8,
                                         /*nlo=*/0, /*nhi=*/0);
  auto sp = sim.SimParams();
  sp.wf1[0] = 1.0f;
  sp.abs[FieldIndex(Node(6, 6, 0, E, X), sp.x, sp.y)] = 2.0f;
  sp.mat[FieldIndex(Node(8, 8, 16, E, X), sp.x, sp.y, sp.z)] = 0.0f;
  sp.zcoeff[10].edz = 3.0f;
  sp.zcoeff[16].hdz = 2.0f;

  sp.mat[FieldIndex(Node(8, 8, 16, E, X), sp.x, sp.y, sp.z)] = 0.0f;

  sim.SimAndTest();
}

TEST(Verification, PointSourceSubvolumeWithModifications) {
  Node srcnode(7, 7, 16, E, Y);
  PointSim<float, float, /*Npml=*/0> sim(
      /*mat0=*/1.0f, srcnode,
      /*timestep=*/8,
      /*nlo=*/0,
      /*nhi=*/0,
      /*srctype=*/RunShape::Src::ZSLICE,
      /*xrange=*/RunShape::Out::Range(2, 14),
      /*yrange=*/RunShape::Out::Range(2, 14),
      /*zrange=*/
      RunShape::Out::Range(5, diamond::ExtZz<float>(/*Npml=*/0) - 5));
  auto sp = sim.SimParams();
  sp.wf1[0] = 1.0f;
  sp.abs[FieldIndex(Node(6, 6, 0, E, X), sp.x, sp.y)] = 2.0f;
  sp.mat[FieldIndex(Node(8, 8, 16, E, X), sp.x, sp.y, sp.z)] = 0.0f;
  sp.zcoeff[10].edz = 3.0f;
  sp.zcoeff[16].hdz = 2.0f;

  sp.mat[FieldIndex(Node(8, 8, 16, E, X), sp.x, sp.y, sp.z)] = 0.0f;

  sim.SimAndTest();
}

TEST(Verification, PointSourceWithPml) {
  constexpr const int nlo = 5;
  constexpr const int nhi = 7;
  Node srcnode(7, 7, 5, E, X);
  PointSim<float, float, /*Npml=*/(nlo + nhi) / Nz> sim(/*mat0=*/1.0f, srcnode,
                                                        /*timestep=*/4, nlo,
                                                        nhi);
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.0f;

  // Modify pml coefficients.
  const reference::ZCoeff<float> zc = {0.2f, 0.3f, 0.5f, 1.0f, 0.4f, 0.6f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  sim.SimAndTest();
}

TEST(Verification, PointSourceSubvolumeWithPml) {
  constexpr const int nlo = 5;
  constexpr const int nhi = 7;
  constexpr const int Npml = (nlo + nhi) / Nz;
  Node srcnode(7, 7, 5, E, X);
  PointSim<float, float, Npml> sim(
      /*mat0=*/1.0f, srcnode,
      /*timestep=*/4, nlo, nhi,
      /*srctype=*/RunShape::Src::ZSLICE,
      /*xrange=*/RunShape::Out::Range(2, 14),
      /*yrange=*/RunShape::Out::Range(2, 14),
      /*zrange=*/
      RunShape::Out::Range(5, diamond::ExtZz<float>(Npml) - 5));
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.0f;

  // Modify pml coefficients.
  const reference::ZCoeff<float> zc = {0.2f, 0.3f, 0.5f, 1.0f, 0.4f, 0.6f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  sim.SimAndTest();
}

TEST(Verification, FloatSim) {
  Node srcnode(7, 7, 17, E, X);
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/0.5f, srcnode,
                                         /*timestep=*/8, /*nlo=*/0,
                                         /*nhi=*/0);
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.5f;
  sim.SimAndTest();
}

// Note that precision starts to diverge between
// fp16 (half2) and fp32 (float) values after timestep 4.
TEST(Verification, Half2Sim) {
  Node srcnode(7, 7, 15, E, Y);
  PointSim<half2, float, /*Npml=*/0> sim(/*mat0=*/0.5f, srcnode,
                                         /*timestep=*/4, /*nlo=*/0,
                                         /*nhi=*/0);
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.5f;
  sim.SimAndTest();
}

TEST(Verification, Half2SimSubvolume) {
  Node srcnode(7, 7, 15, E, Y);
  PointSim<half2, float, /*Npml=*/0> sim(
      /*mat0=*/0.5f, srcnode,
      /*timestep=*/4,
      /*nlo=*/0,
      /*nhi=*/0,
      /*srctype=*/RunShape::Src::ZSLICE,
      /*xrange=*/RunShape::Out::Range(2, 15),
      /*yrange=*/RunShape::Out::Range(1, 14),
      /*zrange=*/
      RunShape::Out::Range(10, diamond::ExtZz<half2>(/*Npml=*/0) - 10));
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.5f;
  sim.SimAndTest();
}

TEST(Verification, Half2WithModifications) {
  Node srcnode(7, 7, 16, E, Y);
  PointSim<half2, float, /*Npml=*/0> sim(/*mat0=*/0.5f, srcnode,
                                         /*timestep=*/4, /*nlo=*/0,
                                         /*nhi=*/0);
  auto sp = sim.SimParams();
  sp.wf1[0] = 0.5f;
  sp.abs[FieldIndex(Node(6, 6, 0, E, X), sp.x, sp.y)] = 2.0f;
  sp.mat[FieldIndex(Node(8, 8, 16, E, X), sp.x, sp.y, sp.z)] = 0.0f;
  sp.zcoeff[13].edz = 2.0f;
  sp.zcoeff[14].hdz = 0.5f;
  sim.SimAndTest();
}

TEST(Verification, Half2WithPml) {
  constexpr const int nlo = 12;
  constexpr const int nhi = 12;
  Node srcnode(7, 7, 10, E, Y); // Completely in bottom pml.
  PointSim<half2, float, /*Npml=*/(nlo + nhi) / diamond::EffNz<half2>()> sim(
      /*mat0=*/0.5f, srcnode, /*timestep=*/2, nlo, nhi);
  auto sp = sim.SimParams();
  sp.wf0[0] = 0.5f;

  // Modify pml coefficients.
  const reference::ZCoeff<float> zc = {2.0f, 1.0f, 0.5f, 1.0f, 2.0f, 0.25f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  sim.SimAndTest();
}

TEST(Verification, Half2SubvolumeWithPml) {
  constexpr const int nlo = 12;
  constexpr const int nhi = 12;
  constexpr const int Npml = (nlo + nhi) / diamond::EffNz<half2>();
  Node srcnode(7, 7, 10, E, Y); // Completely in bottom pml.
  PointSim<half2, float, Npml> sim(
      /*mat0=*/0.5f, srcnode, /*timestep=*/2, nlo, nhi,
      /*srctype=*/RunShape::Src::ZSLICE,
      /*xrange=*/RunShape::Out::Range(2, 14),
      /*yrange=*/RunShape::Out::Range(3, 15),
      /*zrange=*/
      RunShape::Out::Range(2, diamond::ExtZz<half2>(Npml) - 1));
  auto sp = sim.SimParams();
  sp.wf0[0] = 0.5f;

  // Modify pml coefficients.
  const reference::ZCoeff<float> zc = {2.0f, 1.0f, 0.5f, 1.0f, 2.0f, 0.25f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  sim.SimAndTest();
}

TEST(Verification, PointSourceYSliceX) {
  const Node srcnode(6, 6, 17, E, X);
  const int timestep = 8;
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/1.0f, srcnode, timestep,
                                         /*nlo=*/0,
                                         /*nhi=*/0,
                                         /*srctype=*/RunShape::Src::YSLICE);
  auto sp = sim.SimParams();
  for (int t = 0; t < timestep; ++t)
    sp.wf0[t] = 1.0f;
  sim.SimAndTest();
}

TEST(Verification, PointSourceYSliceXAlt) {
  const Node srcnode(6, 5, 17, E, X);
  const int timestep = 8;
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/1.0f, srcnode, timestep,
                                         /*nlo=*/0,
                                         /*nhi=*/0,
                                         /*srctype=*/RunShape::Src::YSLICE);
  auto sp = sim.SimParams();
  for (int t = 0; t <= timestep; ++t)
    sp.wf1[t] = 1.0f;
  sim.SimAndTest();
}

TEST(Verification, PointSourceYSliceZAlt) {
  const Node srcnode(6, 5, 17, E, Z);
  const int timestep = 8;
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/1.0f, srcnode, timestep,
                                         /*nlo=*/0,
                                         /*nhi=*/0,
                                         /*srctype=*/RunShape::Src::YSLICE);
  auto sp = sim.SimParams();
  for (int t = 0; t <= timestep; ++t)
    sp.wf1[t] = 1.0f;
  sim.SimAndTest();
}

TEST(Verification, PointSourceYSliceZ) {
  const Node srcnode(6, 6, 17, E, Z);
  const int timestep = 8;
  PointSim<float, float, /*Npml=*/0> sim(/*mat0=*/1.0f, srcnode, timestep,
                                         /*nlo=*/0,
                                         /*nhi=*/0,
                                         /*srctype=*/RunShape::Src::YSLICE);
  auto sp = sim.SimParams();
  for (int t = 0; t <= timestep; ++t)
    sp.wf0[t] = 1.0f;
  sim.SimAndTest();
}

TEST(Verification, YSrcWithPml) {
  constexpr const int nlo = 12;
  constexpr const int nhi = 2;
  Node srcnode(10, 6, 10, E, X); // Completely in bottom pml.
  PointSim<float, float, /*Npml=*/(nlo + nhi) / diamond::EffNz<float>()> sim(
      /*mat0=*/0.5f, srcnode, /*timestep=*/4, nlo, nhi,
      /*srctype=*/RunShape::Src::YSLICE);
  auto sp = sim.SimParams();
  sp.wf0[0] = 0.5f;

  // Modify pml coefficients.
  const reference::ZCoeff<float> zc = {2.0f, 1.0f, 0.5f, 1.0f, 0.25f, 0.5f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  sim.SimAndTest();
}

TEST(Verification, YSrcHalf2WithPml) {
  constexpr const int nlo = 12;
  constexpr const int nhi = 12;
  Node srcnode(10, 6, 10, E, X); // Completely in bottom pml.
  PointSim<half2, float, /*Npml=*/(nlo + nhi) / diamond::EffNz<half2>()> sim(
      /*mat0=*/0.5f, srcnode, /*timestep=*/2, nlo, nhi,
      /*srctype=*/RunShape::Src::YSLICE);
  auto sp = sim.SimParams();
  sp.wf0[0] = 0.5f;

  // Modify pml coefficients.
  const reference::ZCoeff<float> zc = {2.0f, 1.0f, 0.5f, 1.0f, 1.5f, 0.5f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  sim.SimAndTest();
}

template <typename T> T Energy(T *ptr, int n) {
  T sum = Zero<T>();
  for (int i = 0; i < n; ++i)
    sum += ptr[i];
  return sum;
}

template <typename T, typename T1>
T1 EnergyAtTimestep(int timestep, RunShape::Src::Type srctype) {
  constexpr const int nlo = 10;
  constexpr const int nhi = 10;
  constexpr const int npml = (nlo + nhi) / diamond::EffNz<T>();
  Node srcnode(8, 8, diamond::ExtZz<T>(npml) / 2, E, X);
  PointSim<T, T1, /*Npml=*/npml> sim(/*mat0=*/0.577f, srcnode, timestep, nlo,
                                     nhi, srctype);
  auto sp = sim.SimParams();
  sp.wf0[0] = 1.0f;

  // Modify pml coefficients.
  const reference::ZCoeff<T1> zc = {1.0f, -0.3f, 0.7f, 1.0f, -0.3f, 0.7f};
  for (int i = 0; i < nlo; ++i)
    sp.zcoeff[i] = zc;
  for (int i = 0; i < nhi; ++i)
    sp.zcoeff[sp.z - i - 1] = zc;

  int numelems = reference::FieldElems(sp.x, sp.y, sp.z);
  testutils::Array<T1> out(numelems);
  sim.Sim(out.Ptr());
  return Energy(out.Ptr(), numelems);
}

// Ensures that the simulation values do not blow up.
TEST(Verification, Stability) {
  EXPECT_LT((EnergyAtTimestep<float, float>(1000, RunShape::Src::ZSLICE)),
            10.0f);
  EXPECT_LT((EnergyAtTimestep<float, float>(1000, RunShape::Src::YSLICE)),
            10.0f);
  EXPECT_LT((EnergyAtTimestep<half2, float>(1000, RunShape::Src::ZSLICE)),
            10.0f);
  EXPECT_LT((EnergyAtTimestep<half2, float>(1000, RunShape::Src::YSLICE)),
            10.0f);
}

} // namespace
} // namespace verification
