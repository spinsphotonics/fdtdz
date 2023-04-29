#include <gtest/gtest.h>

#include "buffer.h"
#include "testutils.h"

namespace buffer {
namespace {

using defs::kWarpSize;
using defs::RunShape;
using defs::UV;
using defs::XY;
using diamond::E;
using diamond::H;
using diamond::N;
using diamond::Node;
using diamond::Nz;

Node AbsNode(Node n, UV warppos) {
  UV w = warppos * (N / 2);
  return Node(n.i - w.u - w.v, n.j - w.u + w.v, n.k, n.ehc, n.xyz);
}

int AbsNodeHash(Node node, int threadpos, UV warppos) {
  Node n = AbsNode(node, warppos);
  return diamond::Index(n.xyz) +
         10 * (diamond::Index(n.ehc) +
               10 * (n.k + 100 * (n.j + 100 * (n.i + 100 * threadpos))));
}

// `true` iff `node` also corresponds to a trailing node from a different warp,
// implicitly eliminates C nodes which have no overlap.
bool IsTrailingAbsNode(Node node, UV warppos, UV blockshape) {
  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (Node n : diamond::AllNodes)
        if (diamond::IsTrailingEdge(n))
          if (UV(wu, wv) != warppos && n != node)
            if (AbsNode(node, warppos) == AbsNode(n, UV(wu, wv)))
              return true;
  return false;
}

// Test that we can pass values within a block.
void SharedBufferCoverage(UV blockshape) {
  int nelems = SharedElems(RunShape(blockshape,
                                    /*gridshape=*/UV(1, 1),
                                    /*spacing=*/1,
                                    /*domain=*/XY(0, 0)));
  testutils::Array<int> array(nelems);

  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos) {
        // Populate cell.
        Cell<int> cell;
        diamond::InitCell(cell, 42);

        // Store values.
        StoreShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, E);
        StoreShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, H);
      }

  for (int i = 0; i < nelems; ++i)
    ASSERT_EQ(array[i], 42) << i;
}

TEST(Buffer, SharedBufferCoverage) { SharedBufferCoverage(UV(2, 2)); }

// Test that we can pass values within a block.
void LoadSharedBufferCoverage(UV blockshape) {
  int nelems = SharedElems(RunShape(blockshape,
                                    /*gridshape=*/UV(1, 1),
                                    /*spacing=*/1,
                                    /*domain=*/XY(0, 0)));
  testutils::Array<int> array(nelems);
  for (int i = 0; i < nelems; ++i)
    array[i] = i + 1;

  std::set<int> inds;
  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos) {
        // Populate cell.
        Cell<int> cell;
        diamond::InitCell(cell, -1);

        // Load values.
        LoadShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, E);
        LoadShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, H);

        for (Node n : diamond::AllNodes)
          if (diamond::IsActive(n))
            if (cell.Get(n) != -1) {
              int val = cell.Get(n);
              EXPECT_TRUE(val >= 1 && val <= nelems)
                  << "(val, node, warppos) = (" << val << ", " << n << ", "
                  << UV(wu, wv) << ")";
              inds.insert(cell.Get(n));
            }
      }

  EXPECT_EQ(inds.size(), nelems);
  EXPECT_EQ(*inds.begin(), 1);
  EXPECT_EQ(*inds.rbegin(), nelems);
}

TEST(Buffer, LoadSharedBufferCoverage) { LoadSharedBufferCoverage(UV(1, 2)); }

// Test that we can pass values within a block.
void SharedBufferTransferTest(UV blockshape) {
  int nelems = SharedElems(RunShape(blockshape,
                                    /*gridshape=*/UV(1, 1),
                                    /*spacing=*/1,
                                    /*domain=*/XY(0, 0)));
  testutils::Array<int> array(nelems);

  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos) {
        // Populate cell.
        Cell<int> cell;
        diamond::InitCell(cell, -3);
        for (Node n : diamond::AllNodes)
          if (diamond::IsActive(n))
            cell.Set(diamond::Index(n.xyz) +
                         10 * (diamond::Index(n.ehc) +
                               10 * (n.k + 10 * (n.j + 10 * (n.i + 10 * wv)))),
                     n);

        for (Node n : diamond::AllNodes)
          if (diamond::IsTrailingEdge(n))
            cell.Set(AbsNodeHash(n, threadpos, UV(wu, wv)), n);

        // Store values.
        StoreShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, E);
        StoreShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, H);
      }

  for (int wu = 0; wu < blockshape.u; ++wu)
    for (int wv = 0; wv < blockshape.v; ++wv)
      for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos) {
        // Populate cell.
        Cell<int> cell;
        diamond::InitCell(cell, -2);
        LoadShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, E);
        LoadShared(cell, array.Ptr(), threadpos, UV(wu, wv), blockshape, H);

        // Check values.
        //
        // Note that C nodes have no overlap, and are therefore not checked in
        // this test (which is what we want to happen) since
        // `IsTrailingAbsNode()` will always be false for them.
        //
        for (Node n : diamond::AllNodes)
          if (diamond::IsLeadingEdge(n) &&
              IsTrailingAbsNode(n, UV(wu, wv), blockshape))
            EXPECT_EQ(cell.Get(n), AbsNodeHash(n, threadpos, UV(wu, wv)))
                << "(node, threadpos, warppos, blockshape) = (" << n << ", "
                << threadpos << ", " << UV(wu, wv) << ", " << blockshape << ")";
      }
}

TEST(SharedBuffer, Transfer) {
  SharedBufferTransferTest(/*blockshape=*/UV(3, 4));
}

// Global hash that is based on trailing nodes.
int GlobalHash(Node n, int threadpos, int warppos, int blockpos, int bufpos,
               int blockshape, int gridshape) {
  int j = n.j + (diamond::N / 2) * (warppos + blockshape * blockpos);
  int jj = (diamond::N / 2) * blockshape * gridshape;
  return j + jj * (threadpos +
                   defs::kWarpSize *
                       (n.k + diamond::Nz * (diamond::Index(n.ehc) +
                                             diamond::kNumEhc *
                                                 (diamond::Index(n.xyz) +
                                                  diamond::kNumXyz * bufpos))));
}

// TODO: Test Global buffer by first writing to the entire buffer with a global
// hash and then reading from the whole buffer, making sure to pick up the same
// hash. Do this separately for u- and v-edges.
void GlobalBufferTransferTest(RunShape rs) {
  ASSERT_TRUE(scanner::IsValidRunShape(rs));

  int n = GlobalElems(rs);
  testutils::Array<int> array(n);
  UV buffershape = scanner::BufferShape(rs);

  // Write all block-level u-edges.
  for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos)
    for (int bufpos = 0; bufpos < buffershape.u; ++bufpos)
      for (int blockpos = 0; blockpos < rs.grid.v; ++blockpos)
        for (int warppos = 0; warppos < rs.block.v; ++warppos) {
          Cell<int> cell;
          diamond::InitCell(cell, -42);
          for (Node n : diamond::AllNodes)
            if (diamond::IsActive(n))
              // Add `N / 2` to avoid negative j-values for the trailing u-edge.
              cell.Set(GlobalHash(n, threadpos, warppos, blockpos, bufpos,
                                  rs.block.v, rs.grid.v) +
                           (diamond::N / 2),
                       n);
          StoreGlobalU(cell, array.Ptr(), threadpos, UV(0, warppos),
                       UV(0, blockpos), UV(bufpos, 0), rs, E);
          StoreGlobalU(cell, array.Ptr(), threadpos, UV(0, warppos),
                       UV(0, blockpos), UV(bufpos, 0), rs, H);
        }

  // Write all block-level v-edges.
  for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos)
    for (int bufpos = 0; bufpos < buffershape.v; ++bufpos)
      for (int blockpos = 0; blockpos < rs.grid.u; ++blockpos)
        for (int warppos = 0; warppos < rs.block.u; ++warppos) {
          Cell<int> cell;
          diamond::InitCell(cell, -42);
          for (Node n : diamond::AllNodes)
            if (diamond::IsActive(n))
              // Note that warp-level indices must be reversed because of the
              // need to store the entire edge in ascending `j` order.
              cell.Set(GlobalHash(n, threadpos, rs.block.u - warppos - 1,
                                  blockpos, bufpos, rs.block.u, rs.grid.u),
                       n);
          StoreGlobalV(cell, array.Ptr(), threadpos, UV(warppos, 0),
                       UV(blockpos, 0), UV(0, bufpos), rs, E);
          StoreGlobalV(cell, array.Ptr(), threadpos, UV(warppos, 0),
                       UV(blockpos, 0), UV(0, bufpos), rs, H);
        }

  // Check all block-level u-edges.
  for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos)
    for (int bufpos = 0; bufpos < buffershape.u; ++bufpos)
      for (int blockpos = 0; blockpos < rs.grid.v; ++blockpos)
        for (int warppos = 0; warppos < rs.block.v; ++warppos) {
          Cell<int> cell;
          diamond::InitCell(cell, -42);
          LoadGlobalU(cell, array.Ptr(), threadpos, UV(0, warppos),
                      UV(0, blockpos), UV(bufpos, 0), rs, E);
          LoadGlobalU(cell, array.Ptr(), threadpos, UV(0, warppos),
                      UV(0, blockpos), UV(bufpos, 0), rs, H);

          for (Node n : diamond::AllNodes)
            if (diamond::IsLeadingEdge(n) && n.ehc != C)
              // Bottom-point is not included for E- of H-field u-edges.
              if (n.j > 0)
                ASSERT_EQ(cell.Get(n), //
                          GlobalHash(n, threadpos, warppos, blockpos, bufpos,
                                     rs.block.v, rs.grid.v))
                    << "u-edge at (node, threadpos, warppos, blockpos, bufpos, "
                       "blockshape, gridshape, bufshape) = ("
                    << n << ", " << threadpos << ", " << warppos << ", "
                    << blockpos << ", " << bufpos << ", " << rs.block << ", "
                    << rs.grid << ", " << scanner::BufferShape(rs) << ")";
        }

  // Check all block-level v-edges.
  for (int threadpos = 0; threadpos < defs::kWarpSize; ++threadpos)
    for (int bufpos = 0; bufpos < buffershape.v; ++bufpos)
      for (int blockpos = 0; blockpos < rs.grid.u; ++blockpos)
        for (int warppos = 0; warppos < rs.block.u; ++warppos) {
          Cell<int> cell;
          diamond::InitCell(cell, -42);
          LoadGlobalV(cell, array.Ptr(), threadpos, UV(warppos, 0),
                      UV(blockpos, 0), UV(0, bufpos), rs, E);
          LoadGlobalV(cell, array.Ptr(), threadpos, UV(warppos, 0),
                      UV(blockpos, 0), UV(0, bufpos), rs, H);

          for (Node n : diamond::AllNodes)
            if (diamond::IsLeadingEdge(n))
              // Only the v-edge for H-values includes the leading point.
              if (n.ehc != C &&
                  (n.j == 0 || n.j == -1 ||
                   (n.j == -2 && (n.ehc == H) &&
                    (n.xyz == diamond::X || n.xyz == diamond::Z))))
                ASSERT_EQ(cell.Get(n),
                          //
                          // Add `N / 2` to avoid negative j-values on the
                          // leading v-edge.
                          //
                          // Note that warp-level indices must be reversed
                          // because of the need to store the entire edge in
                          // ascending `j` order.
                          GlobalHash(n, threadpos, rs.block.u - warppos - 1,
                                     blockpos, bufpos, rs.block.u, rs.grid.u) +
                              (diamond::N / 2))
                    << "v-edge at (node, threadpos, warppos, blockpos, bufpos, "
                       "blockshape, gridshape, bufshape) = ("
                    << n << ", " << threadpos << ", " << warppos << ", "
                    << blockpos << ", " << bufpos << ", " << rs.block << ", "
                    << rs.grid << ", " << scanner::BufferShape(rs) << ")";
        }
}

TEST(GlobalBuffer, Transfer) {
  GlobalBufferTransferTest(RunShape(/*blockshape=*/UV(3, 4),
                                    /*gridshape=*/UV(5, 8),
                                    /*blockspacing=*/2,
                                    /*domain=*/XY(100, 94)));
}

} // namespace
} // namespace buffer
