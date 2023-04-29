#include <gtest/gtest.h>

#include "defs.h"
#include "diamond.h"
#include "scanner.h"
#include "testutils.h"

namespace scanner {
namespace {

using diamond::Ehc;
using diamond::Node;
using diamond::Xyz;

// TEST(BufferShape, IndividualCases) {
//   // Smallest case.
//   EXPECT_EQ(BufferShape(RunShape(/*blockshape=*/UV(1, 1),
//                                  /*gridshape=*/UV(1, 1),
//                                  /*spacing=*/0,
//                                  /*domain=*/XY(2, 4))),
//             UV(0, 0));
//
//   // Single-pass case.
//   EXPECT_EQ(BufferShape(RunShape(/*blockshape=*/UV(2, 4),
//                                  /*gridshape=*/UV(8, 5),
//                                  /*spacing=*/0,
//                                  /*domain=*/XY(100, 72))),
//             UV(68, 60));
//
//   // General case.
//   EXPECT_EQ(BufferShape(RunShape(/*blockshape=*/UV(2, 4),
//                                  /*gridshape=*/UV(8, 5),
//                                  /*spacing=*/0,
//                                  /*domain=*/XY(1000, 432))),
//             UV(9968, 960));
// }
//
// // Tests that the sum of all nodes in the diamond (local register storage)
// and
// // the corresponding buffer is exactly equal to the number of nodes in the
// // simulation domain. That is, that we can store the simulation in the
// // combined diamond and buffer structures.
// void BufferAndDiamondCoverAllDomainNodesTest(RunShape rs) {
//   UV dshape = Diamond(rs);
//   UV bufshape = BufferShape(rs);
//   EXPECT_EQ(2 * Diamond(rs).u * Diamond(rs).v +
//                 BufferShape(rs).u * Diamond(rs).v +
//                 BufferShape(rs).v * Diamond(rs).u,
//             rs.domain.x * rs.domain.y)
//       << "(dshape, bufshape, domain) = (" << dshape << ", " << bufshape << ",
//       "
//       << rs.domain << ")";
// }
//
// TEST(BufferShape, BufferAndDiamondCoverAllDomainNodes) {
//   BufferAndDiamondCoverAllDomainNodesTest(RunShape(/*blockshape=*/UV(1, 1),
//                                                    /*gridshape=*/UV(1, 1),
//                                                    /*blockspacing=*/0,
//                                                    /*domain=*/XY(2, 4)));
//
//   BufferAndDiamondCoverAllDomainNodesTest(RunShape(/*blockshape=*/UV(2, 4),
//                                                    /*gridshape=*/UV(8, 5),
//                                                    /*blockspacing=*/0,
//                                                    /*domain=*/XY(1000,
//                                                    432)));
// }
//
// TEST(IsValidRunShape, InvidividualCases) {
//   // Smallest case.
//   EXPECT_TRUE(IsValidRunShape(RunShape(/*blockshape=*/UV(1, 1),
//                                        /*gridshape=*/UV(1, 1),
//                                        /*spacing=*/0,
//                                        /*domain=*/XY(2, 4))));
//
//   // Smallest case should fail if spacing is added.
//   EXPECT_FALSE(IsValidRunShape(RunShape(/*blockshape=*/UV(1, 1),
//                                         /*gridshape=*/UV(1, 1),
//                                         /*spacing=*/1,
//                                         /*domain=*/XY(2, 4))));
//
//   // General case, that barely fits in terms of buffer shape.
//   EXPECT_TRUE(IsValidRunShape(RunShape(/*blockshape=*/UV(2, 4),
//                                        /*gridshape=*/UV(8, 5),
//                                        /*spacing=*/2,
//                                        /*domain=*/XY(50, 72))));
//
//   // General case that barely does not fit.
//   EXPECT_FALSE(IsValidRunShape(RunShape(/*blockshape=*/UV(2, 4),
//                                         /*gridshape=*/UV(8, 5),
//                                         /*spacing=*/2,
//                                         /*domain=*/XY(49, 72))));
//
//   // Accomodates 1 diamond u-extent and 10 diamond v-extents.
//   EXPECT_TRUE(IsValidRunShape(RunShape(/*blockshape=*/UV(2, 4),
//                                        /*gridshape=*/UV(8, 5),
//                                        /*spacing=*/2,
//                                        /*domain=*/XY(1000, 432))));
//
//   // Does not accomodate an integer multiple of v-extents.
//   EXPECT_FALSE(IsValidRunShape(RunShape(/*blockshape=*/UV(2, 4),
//                                         /*gridshape=*/UV(8, 5),
//                                         /*spacing=*/2,
//                                         /*domain=*/XY(1000, 431))));
//
//   // Works, not u-dominant.
//   EXPECT_TRUE(IsValidRunShape(RunShape(/*blockshape=*/UV(2, 3),
//                                        /*gridshape=*/UV(3, 2),
//                                        /*spacing=*/2,
//                                        /*domain=*/XY(1000, 24))));
//
//   // Fails because it is not v-dominant.
//   EXPECT_FALSE(IsValidRunShape(RunShape(/*blockshape=*/UV(2, 3),
//                                         /*gridshape=*/UV(4, 2),
//                                         /*spacing=*/2,
//                                         /*domain=*/XY(1000, 28))));
// }

void ScanDomain(int timesteps, RunShape rs) {
  int steps = NumSteps(timesteps, rs);
  std::cout << "(timesteps, steps, rs) = (" << timesteps << ", " << steps
            << ", " << rs << ")\n";
  int numelems = diamond::kNumXyz * rs.domain.x * rs.domain.y;
  testutils::Array<int> array(numelems);

  for (int bu = 0; bu < rs.grid.u; ++bu)
    for (int bv = 0; bv < rs.grid.v; ++bv)
      for (int wu = 0; wu < rs.block.u; ++wu)
        for (int wv = 0; wv < rs.block.v; ++wv)
          for (int step = 0; step < steps; ++step) {
            XYT domainpos = DomainPos(step, UV(wu, wv), UV(bu, bv), rs);
            if (domainpos.t >= timesteps)
              for (Node n : diamond::AllNodes)
                if (diamond::IsInsideDiamond(n) && n.k == 0 &&
                    n.ehc == diamond::E) {
                  XY pos(domainpos.x + n.i, domainpos.y + n.j);
                  if (pos.x >= 0 && pos.x < rs.domain.x && //
                      pos.y >= 0 && pos.y < rs.domain.y) {
                    array[pos.x +
                          rs.domain.x *
                              (pos.y + rs.domain.y * diamond::Index(n.xyz))] =
                        domainpos.t;
                  }
                }
          }

  int cnt = 0;
  int max = 0;
  for (int i = 0; i < numelems; ++i) {
    if (array[i] < timesteps)
      ++cnt;
    if (array[i] > max)
      max = array[i];
  }
  EXPECT_EQ(cnt, 0);
  // Don't do more work than this.
  EXPECT_LE(max, timesteps + 2 * (rs.domain.x + rs.domain.y));
}

TEST(NumSteps, ScansEntireDomain) {
  // TODO: Run this on shapes that don't work first. Hint: reference the colab.
  ScanDomain(/*timesteps=*/1010, RunShape(UV(2, 2), UV(2, 2), 2, XY(40, 40)));
  ScanDomain(/*timesteps=*/1500, RunShape(UV(2, 4), UV(6, 6), 2, XY(216, 216)));
}

// TEST(BufferPos, InidividualCases) {
//   RunShape rs = RunShape(/*blockshape=*/UV(2, 4),
//                          /*gridshape=*/UV(8, 5),
//                          /*blockspacing=*/2,
//                          /*domain=*/XY(1000, 432));
//
//   // First test the distribution of blocks at step 0.
//   EXPECT_EQ(BufferPos(/*step=*/0, /*blockpos=*/UV(0, 0), rs), UV(14, 8));
//   EXPECT_EQ(BufferPos(/*step=*/0, /*blockpos=*/UV(0, 1), rs), UV(14, 6));
//   EXPECT_EQ(BufferPos(/*step=*/0, /*blockpos=*/UV(1, 0), rs), UV(12, 8));
//   EXPECT_EQ(BufferPos(/*step=*/0, /*blockpos=*/UV(7, 4), rs), UV(0, 0));
//
//   // Test that each step advances one's place in the buffers.
//   EXPECT_EQ(BufferPos(/*step=*/1, /*blockpos=*/UV(0, 0), rs), UV(15, 9));
//
//   // Test wrap-around.
//   EXPECT_EQ(BufferPos(/*step=*/960, /*blockpos=*/UV(7, 4), rs), UV(960, 0));
// }
//
// TEST(DomainPos, IndividualCases) {
//   { // Single diamond.
//     RunShape rs(/*blockshape=*/UV(1, 1),
//                 /*gridshape=*/UV(1, 1),
//                 /*spacing=*/0,
//                 /*domain=*/XY(10, 10));
//
//     // Test the first few steps.
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(0, 2, -14));
//     EXPECT_EQ(DomainPos(/*step=*/1, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(1, 2, -13));
//     EXPECT_EQ(DomainPos(/*step=*/2, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(2, 2, -12));
//
//     // Test that the "connection" between step 2 and 40 works (they are at
//     the
//     // same time step).
//     EXPECT_EQ(DomainPos(/*step=*/40, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(0, 0, -12));
//   }
//
//   { // Single 2x2 block.
//     RunShape rs(/*blockshape=*/UV(2, 2),
//                 /*gridshape=*/UV(1, 1),
//                 /*spacing=*/0,
//                 /*domain=*/XY(10, 10));
//
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(4, 4, -8));
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 1), /*b=*/UV(0, 0), rs),
//               XYT(2, 6, -8));
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(1, 0), /*b=*/UV(0, 0), rs),
//               XYT(2, 2, -8));
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(1, 1), /*b=*/UV(0, 0), rs),
//               XYT(0, 4, -8));
//   }
//
//   { // 2x2 grid with 1x1 blocks and non-zero blockspacing.
//     RunShape rs(/*blockshape=*/UV(1, 1),
//                 /*gridshape=*/UV(2, 2),
//                 /*spacing=*/2,
//                 /*domain=*/XY(10, 10));
//
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(8, 4, -4));
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 0), /*b=*/UV(0, 1), rs),
//               XYT(4, 6, -6));
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 0), /*b=*/UV(1, 0), rs),
//               XYT(4, 2, -6));
//     EXPECT_EQ(DomainPos(/*step=*/0, /*w=*/UV(0, 0), /*b=*/UV(1, 1), rs),
//               XYT(0, 4, -8));
//   }
//
//   { // Single 2x1 block.
//     RunShape rs(/*blockshape=*/UV(2, 1),
//                 /*gridshape=*/UV(1, 1),
//                 /*spacing=*/0,
//                 /*domain=*/XY(10, 10));
//
//     // Test the connection.
//     EXPECT_EQ(DomainPos(/*step=*/2, /*w=*/UV(1, 0), /*b=*/UV(0, 0), rs),
//               XYT(2, 2, -8));
//     EXPECT_EQ(DomainPos(/*step=*/28, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(0, 0, -8));
//   }
//
//   { // Single 2x1 grid with 1x1 block.
//     RunShape rs(/*blockshape=*/UV(1, 1),
//                 /*gridshape=*/UV(2, 1),
//                 /*spacing=*/3,
//                 /*domain=*/XY(10, 10));
//
//     // Test the connection with non-zero spacing.
//     EXPECT_EQ(DomainPos(/*step=*/2, /*w=*/UV(0, 0), /*b=*/UV(1, 0), rs),
//               XYT(2, 2, -8));
//     EXPECT_EQ(DomainPos(/*step=*/25, /*w=*/UV(0, 0), /*b=*/UV(0, 0), rs),
//               XYT(0, 0, -8));
//   }
// }
//
// int DomainIndex(int x, int y, Ehc ehc, Xyz xyz, RunShape rs) {
//   return x + rs.domain.x *
//                  (y + rs.domain.y * (diamond::Index(ehc) +
//                                      diamond::kNumEhc *
//                                      diamond::Index(xyz)));
// }
//
// bool IsEligible(int x, int y, RunShape rs) {
//   int pad = diamond::N / 2;
//   return x >= pad && x < rs.domain.x - pad && y >= pad && y < rs.domain.y -
//   pad;
// }
//
// bool IsEligible(XYT pos, Node n, RunShape rs) {
//   return IsEligible(pos.x + n.i, pos.y + n.j, rs);
// }
//
// bool IsLowerHalf(Node n) { return n.j <= 0; }
//
// bool IsUpperHalf(Node n) {
//   return n.j > 0 ||
//          (n.j == 0 && ((n.ehc == diamond::H) != (n.xyz == diamond::Y)));
// }
//
// // Ensures that scanning is self-consistent on the simulation domain.
// void CheckTimeValues(int *array, int step, UV w, UV b, RunShape rs) {
//   XYT pos = DomainPos(step, w, b, rs);
//   for (Node n : diamond::AllNodes)
//     if (diamond::IsActive(n) && !diamond::IsInsideDiamond(n) && n.k == 0 &&
//         pos.t >= 0 && IsEligible(pos, n, rs) &&
//         ((w.u == 0 && IsUpperHalf(n)) || (w.v == 0 && IsLowerHalf(n))))
//       EXPECT_EQ(array[DomainIndex(pos.x + n.i, pos.y + n.j, n.ehc, n.xyz,
//       rs)],
//                 pos.t)
//           << "(step, warppos, blockpos, pos, n) = (" << step << ", " << w
//           << ", " << b << ", " << pos << ", " << n << ")";
// }
//
// void WriteTimeValues(int *array, int step, UV w, UV b, RunShape rs) {
//   XYT pos = DomainPos(step, w, b, rs);
//   for (Node n : diamond::AllNodes)
//     if (diamond::IsTrailingEdge(n) && n.k == 0 && pos.t >= 0 &&
//         IsEligible(pos, n, rs) &&
//         ((w.u == rs.block.u - 1 && IsLowerHalf(n)) ||
//          (w.v == rs.block.v - 1 && IsUpperHalf(n))))
//       array[DomainIndex(pos.x + n.i, pos.y + n.j, n.ehc, n.xyz, rs)] = pos.t;
// }
//
// // Ensures that scanning is self-consistent on the simulation domain.
// //
// // Verifies that the values loaded into the diamond are always consistent
// with
// // the time step of the diamond.
// //
// void DomainPosSelfConsistencyTest(RunShape rs, int numsteps) {
//   int n = rs.domain.x * rs.domain.y * diamond::kNumEhc * diamond::kNumXyz;
//   testutils::Array<int> array(n);
//   for (int i = 0; i < n; ++i)
//     array[i] = 0;
//
//   for (int step = 0; step < numsteps; ++step) {
//     // Check that loaded values are consistent with diamond time step.
//     for (int bu = 0; bu < rs.grid.u; ++bu)
//       for (int bv = 0; bv < rs.grid.v; ++bv)
//         for (int wu = 0; wu < rs.block.u; ++wu)
//           for (int wv = 0; wv < rs.block.v; ++wv)
//             CheckTimeValues(array.Ptr(), step, UV(wu, wv), UV(bu, bv), rs);
//
//     // Write values back to the simulation domain.
//     for (int bu = 0; bu < rs.grid.u; ++bu)
//       for (int bv = 0; bv < rs.grid.v; ++bv)
//         for (int wu = 0; wu < rs.block.u; ++wu)
//           for (int wv = 0; wv < rs.block.v; ++wv)
//             WriteTimeValues(array.Ptr(), step, UV(wu, wv), UV(bu, bv), rs);
//   }
//
//   // Check that all the timestamps are at least at timestamp 0.
//   for (int x = 0; x < rs.domain.x; ++x)
//     for (int y = 0; y < rs.domain.y; ++y)
//       for (Ehc ehc : {diamond::E, diamond::H, diamond::C})
//         for (Xyz xyz : {diamond::X, diamond::Y, diamond::Z})
//           if (IsEligible(x, y, rs))
//             ASSERT_GT(array[DomainIndex(x, y, ehc, xyz, rs)], 0)
//                 << "(x, y, ehc, xyz) = (" << x << ", " << y << ", " << ehc
//                 << ", " << xyz << ")";
// }
//
// TEST(DomainPos, SelfConsistentForSmallestRunShape) {
//   DomainPosSelfConsistencyTest(RunShape(/*blockshape=*/UV(1, 1),
//                                         /*gridshape=*/UV(1, 1),
//                                         /*blockspacing=*/1,
//                                         /*domainshape=*/XY(10, 10)),
//                                /*numsteps=*/250);
// }
//
// TEST(DomainPos, SelfConsistentForMediumRunShape) {
//   DomainPosSelfConsistencyTest(RunShape(/*blockshape=*/UV(2, 2),
//                                         /*gridshape=*/UV(2, 2),
//                                         /*blockspacing=*/1,
//                                         /*domainshape=*/XY(16, 16)),
//                                /*numsteps=*/64);
// }
//
// TEST(DomainPos, SelfConsistentForLargeRunShape) {
//   DomainPosSelfConsistencyTest(RunShape(/*blockshape=*/UV(2, 4),
//                                         /*gridshape=*/UV(8, 5),
//                                         /*blockspacing=*/2,
//                                         /*domainshape=*/XY(80, 80)),
//                                /*numsteps=*/800);
// }

} // namespace
} // namespace scanner
