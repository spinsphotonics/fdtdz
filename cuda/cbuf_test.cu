#include <gtest/gtest.h>

#include "cbuf.h"
#include "diamond.h"
#include "scanner.h"
#include "testutils.h"

namespace cbuf {
namespace {

using defs::IsAux;
using defs::kWarpSize;
using defs::RunShape;
using defs::UV;
using defs::XY;
using defs::XYT;
using diamond::C;
using diamond::N;
using diamond::Node;
using diamond::Nz;
using diamond::X;
using diamond::Xyz;
using diamond::Y;
using diamond::Z;

void GlobalIndexContiguousTest(XY domain) {
  std::set<int> inds;
  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; ++j)
      for (int k = 0; k < kWarpSize * Nz; ++k)
        for (Xyz xyz : {X, Y, Z})
          inds.insert(GlobalIndex(Node(i, j, k, C, xyz), domain));

  int n = GlobalElems(domain);
  EXPECT_EQ(inds.size(), n);
  EXPECT_EQ(*inds.begin(), 0);
  EXPECT_EQ(*inds.rbegin(), n - 1);
}

TEST(CBuf, GlobalIndexingContiguous) {
  GlobalIndexContiguousTest(/*domain=*/XY(5, 6));
}

int GlobalNodeHash(Node n, XY domain) {
  return n.k + (2 * Nz * kWarpSize) *
                   (n.i + domain.x * (n.j + domain.y * diamond::Index(n.xyz)));
}

// TODO: Probably want to just completely redo this test?
void WriteToGlobalTest(XY domain, int npml, int zshift) {
  const int dt = 1;
  RunShape::Vol //
      sub(N, domain.x - N, N, domain.y - N, 0, ExtZz<int>(npml)),
      vol(N, domain.x - N, N, domain.y - N, 0, ExtZz<int>(npml));
  testutils::Array<int> ext(ExternalElems(sub));
  for (int i = sub.x0; i < sub.x1; ++i)
    for (int j = sub.y0; j < sub.y1; ++j)
      for (int k = sub.z0; k < sub.z1; ++k)
        for (Xyz xyz : diamond::AllXyz) {
          Node n(i, j, k, diamond::E, xyz);
          ext[ExternalIndex(n, sub)] = GlobalNodeHash(n, domain);
        }

  testutils::Array<int> abs(slice::ZMask<int>::ExternalElems(domain));
  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; ++j)
      for (Xyz xyz : diamond::AllXyz) {
        XY pos(i, j);
        abs[slice::ZMask<int>::ExternalIndex(pos, xyz, domain)] =
            defs::Zero<int>();
      }

  testutils::Array<int> glb(GlobalElems(domain));
  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int i = 0; i < domain.x; ++i)
      for (int j = 0; j < domain.y; ++j)
        for (int k = 0; k < Nz; ++k)
          for (Xyz xyz : diamond::AllXyz)
            WriteGlobal(ext.Ptr(), glb.Ptr(), Node(i, j, k, diamond::E, xyz),
                        domain, threadpos, npml, zshift, IsAux(threadpos, npml),
                        sub, vol, abs.Ptr(), dt);

  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int i = N; i < domain.x - N; ++i)
      for (int j = N; j < domain.y - N; ++j)
        for (int k = 0; k < Nz; ++k)
          for (Xyz xyz : diamond::AllXyz) {
            Node glbnode(i, j, k + Nz * threadpos, diamond::E, xyz);
            Node extnode(i, j, ExtZIndex<int>(k, threadpos, npml, zshift),
                         diamond::E, xyz);
            int expected = IsAux(threadpos, npml)
                               ? defs::One<int>()
                               : GlobalNodeHash(extnode, domain);
            EXPECT_EQ(glb[GlobalIndex(glbnode, domain)], expected)
                << "glbnode = " << glbnode;
          }
}

TEST(CBuf, WriteToGlobal) {
  WriteToGlobalTest(/*domain=*/XY(3 + 2 * N, 4 + 2 * N), /*npml=*/7,
                    /*zshift=*/10);
}

// Need to make numbers smaller so that they don't exceed the precision limit
// for half2 values.
float Half2GlobalNodeHash(Node n, XY domain) {
  return float(GlobalNodeHash(n, domain) % (1 << 11));
}

void Half2WriteToGlobalTest(XY domain, int npml, int zshift) {
  const float dt = 1.0f;
  RunShape::Vol //
      sub(N, domain.x - N, N, domain.y - N, 0, ExtZz<half2>(npml)),
      vol(N, domain.x - N, N, domain.y - N, 0, ExtZz<half2>(npml));
  testutils::Array<float> ext(ExternalElems(sub));
  for (int i = sub.x0; i < sub.x1; ++i)
    for (int j = sub.y0; j < sub.y1; ++j)
      for (int k = sub.z0; k < sub.z1; ++k)
        for (Xyz xyz : diamond::AllXyz) {
          Node n(i, j, k, diamond::E, xyz);
          ext[ExternalIndex(n, sub)] = Half2GlobalNodeHash(n, domain);
        }

  testutils::Array<float> abs(slice::ZMask<float>::ExternalElems(domain));
  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; ++j)
      for (Xyz xyz : diamond::AllXyz) {
        XY pos(i, j);
        abs[slice::ZMask<float>::ExternalIndex(pos, xyz, domain)] =
            defs::Zero<float>();
      }

  testutils::Array<half2> glb(GlobalElems(domain));
  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int i = 0; i < domain.x; ++i)
      for (int j = 0; j < domain.y; ++j)
        for (int k = 0; k < Nz; ++k)
          for (Xyz xyz : diamond::AllXyz)
            WriteGlobal(ext.Ptr(), glb.Ptr(), Node(i, j, k, diamond::E, xyz),
                        domain, threadpos, npml, zshift, IsAux(threadpos, npml),
                        sub, vol, abs.Ptr(), dt);

  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int i = N; i < domain.x - N; ++i)
      for (int j = N; j < domain.y - N; ++j)
        for (int k = 0; k < Nz; ++k)
          for (Xyz xyz : diamond::AllXyz) {
            Node glbnode(i, j, k + Nz * threadpos, diamond::E, xyz);
            Node extnodelo(i, j, ExtZIndex<half2>(k, threadpos, npml, zshift),
                           diamond::E, xyz);
            Node extnodehi(i, j,
                           ExtZIndex<half2>(k + Nz, threadpos, npml, zshift),
                           diamond::E, xyz);
            float expectedlo = IsAux(threadpos, npml)
                                   ? defs::One<float>()
                                   : Half2GlobalNodeHash(extnodelo, domain);
            float expectedhi = IsAux(threadpos, npml)
                                   ? defs::One<float>()
                                   : Half2GlobalNodeHash(extnodehi, domain);
            EXPECT_EQ(__low2float(glb[GlobalIndex(glbnode, domain)]),
                      expectedlo)
                << "lo: (threadpos, global node, extnodelo) = (" << threadpos
                << ", " << glbnode << ", " << extnodelo << ")\n";
            EXPECT_EQ(__high2float(glb[GlobalIndex(glbnode, domain)]),
                      expectedhi)
                << "hi: (threadpos, global node) = (" << threadpos << ", "
                << glbnode << ", " << extnodehi << ")\n";
          }
}

TEST(CBuf, Half2WriteToGlobal) {
  Half2WriteToGlobalTest(/*domain=*/XY(3 + 2 * N, 4 + 2 * N), /*npml=*/7,
                         /*zshift=*/10);
}

void GlobalLoadTest(XY domain) {
  testutils::Array<int> arr(GlobalElems(domain));

  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; ++j)
      for (int k = 0; k < Nz * kWarpSize; ++k)
        for (Xyz xyz : {X, Y, Z}) {
          Node n(i, j, k, C, xyz);
          arr[GlobalIndex(n, domain)] = GlobalNodeHash(n, domain);
        }

  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; ++j)
      if (diamond::IsDiamondCompletelyInDomain(XY(i, j), domain))
        for (int threadpos = 0; threadpos < kWarpSize; ++threadpos) {
          Cell<int> cell;
          diamond::InitCell(cell, defs::Zero<int>());
          LoadGlobal(cell, arr.Ptr(), threadpos, /*warppos=*/UV(0, 0),
                     /*pos=*/XY(i, j), domain);

          for (Node n : diamond::AllNodes)
            if (diamond::IsLeadingEdge(n) && n.ehc == C)
              EXPECT_EQ(
                  cell.Get(n),
                  GlobalNodeHash(n.dI(i).dJ(j).dK(Nz * threadpos), domain));
        }
}

TEST(CBuf, GlobalLoad) { GlobalLoadTest(/*domain=*/XY(7, 8)); }

void SharedLoadContiguousTest(UV blockshape) {
  int n = SharedElems(blockshape);
  testutils::Array<int> arr(n);
  for (int i = 0; i < n; ++i)
    arr[i] = i + 1;

  std::set<int> inds;
  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (int threadpos = 0; threadpos < kWarpSize; ++threadpos) {
        UV warppos(wu, wv);
        Cell<int> cell;
        diamond::InitCell(cell, -42);
        LoadShared(cell, arr.Ptr(), threadpos, warppos, blockshape);
        for (Node n : diamond::AllNodes)
          if (diamond::IsLeadingEdge(n) && n.ehc == C && cell.Get(n) != -42)
            inds.insert(cell.Get(n));
      }

  EXPECT_EQ(inds.size(), n);
  EXPECT_EQ(*inds.begin(), 1);
  EXPECT_EQ(*inds.rbegin(), n);
}

TEST(CBuf, SharedLoadContiguous) {
  SharedLoadContiguousTest(/*blockshape=*/UV(4, 3));
}

void SharedStoreContiguousTest(UV blockshape) {
  int n = SharedElems(blockshape);
  testutils::Array<int> arr(n);

  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (int threadpos = 0; threadpos < kWarpSize; ++threadpos) {
        UV warppos(wu, wv);
        Cell<int> cell;
        diamond::InitCell(cell, 42);
        StoreShared(cell, arr.Ptr(), threadpos, warppos, blockshape);
      }

  std::set<int> inds;
  for (int i = 0; i < n; ++i)
    if (arr[i] == 42)
      inds.insert(i + 1);

  EXPECT_EQ(inds.size(), n);
  EXPECT_EQ(*inds.begin(), 1);
  EXPECT_EQ(*inds.rbegin(), n);
}

TEST(CBuf, SharedStoreContiguous) {
  SharedStoreContiguousTest(/*blockshape=*/UV(4, 3));
}

void TransferTest(int numsteps, RunShape rs) {
  ASSERT_TRUE(scanner::IsValidRunShape(rs));
  testutils::Array<int> arr(GlobalElems(rs.domain));
  testutils::Array<int> sharr(SharedElems(rs.block));

  for (int i = 0; i < rs.domain.x; ++i)
    for (int j = 0; j < rs.domain.y; ++j)
      for (int k = 0; k < Nz * kWarpSize; ++k)
        for (Xyz xyz : {X, Y, Z}) {
          Node n(i, j, k, C, xyz);
          arr[GlobalIndex(n, rs.domain)] = GlobalNodeHash(n, rs.domain);
        }

  for (int bu = 0; bu < rs.grid.u; ++bu)
    for (int bv = 0; bv < rs.grid.v; ++bv) {
      for (int step = 0; step < numsteps; ++step) {
        for (int wu = 0; wu < rs.block.u; ++wu)
          for (int wv = 0; wv < rs.block.v; ++wv)
            for (int threadpos = 0; threadpos < kWarpSize; ++threadpos) {
              UV blockpos(bu, bv), warppos(wu, wv);
              XYT pos = scanner::DomainPos(step - 1, warppos, blockpos, rs);
              // Fill.
              Cell<int> cell;
              for (Node n : diamond::AllNodes)
                if (diamond::IsInsideDiamond(n) && n.ehc == C)
                  cell.Set(
                      GlobalNodeHash(n.dI(pos.x).dJ(pos.y).dK(Nz * threadpos),
                                     rs.domain),
                      n);
              if (pos.x >= N / 2 && pos.x < rs.domain.x && //
                  pos.y >= N / 2 && pos.y < rs.domain.y - (N / 2))
                StoreShared(cell, sharr.Ptr(), threadpos, warppos, rs.block);
            }

        for (int wu = 0; wu < rs.block.u; ++wu)
          for (int wv = 0; wv < rs.block.v; ++wv) {
            int threadpos = 0;
            // for (int threadpos = 0; threadpos < kWarpSize; ++threadpos) {
            UV blockpos(bu, bv), warppos(wu, wv);
            XYT pos = scanner::DomainPos(step, warppos, blockpos, rs);
            Cell<int> cell;
            diamond::InitCell(cell, -42);

            LoadGlobal(cell, arr.Ptr(), threadpos, warppos, XY(pos.x, pos.y),
                       rs.domain);
            LoadShared(cell, sharr.Ptr(), threadpos, warppos, rs.block);

            for (Node n : diamond::AllNodes)
              if (diamond::IsLeadingEdge(n) && n.ehc == C && //
                  pos.x >= N && pos.x < rs.domain.x - N &&   //
                  pos.y >= N && pos.y < rs.domain.y - N &&   //
                  pos.t >= 0)
                EXPECT_EQ(cell.Get(n), GlobalNodeHash(n.dI(pos.x).dJ(pos.y).dK(
                                                          Nz * threadpos),
                                                      rs.domain))
                    << "(step, pos, node, warppos, blockpos) = (" << step
                    << ", " << pos << ", " << n << ", " << warppos << ", "
                    << blockpos << ")";
          }
      }
    }
}

TEST(CBuf, Transfer) {
  TransferTest(/*numsteps=*/1000, RunShape(/*blockshape=*/UV(3, 4),
                                           /*gridshape=*/UV(2, 2),
                                           /*blockspacing=*/2,
                                           /*domainshape=*/XY(30, 44)));
}

} // namespace
} // namespace cbuf
