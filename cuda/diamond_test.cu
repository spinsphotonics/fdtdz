#include <gtest/gtest.h>
#include <set>

#include "defs.h"
#include "diamond.h"
#include "testutils.h"

namespace diamond {
namespace {

// Checks that the set of nodes for which `fn(Node(i, j, k, ehc, xyz)`
// is true, is equal to the set of `(i, j)` nodes in `expected`.
void NodesMatch(bool (*fn)(Node n), Ehc ehc, Xyz xyz,
                std::set<std::pair<int, int>> expected) {
  std::set<std::pair<int, int>> actual;
  for (Node n : AllNodes)
    if (n.ehc == ehc && n.xyz == xyz && fn(n))
      actual.insert(std::pair<int, int>(n.i, n.j));
  EXPECT_EQ(actual, expected);
}

TEST(Diamond, IsInsideDiamond) {
  NodesMatch(
      &IsInsideDiamond, E, X,
      {{-2, 0}, {-2, 1}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}, {0, 0}, {0, 1}});
  NodesMatch(
      &IsInsideDiamond, E, Y,
      {{-2, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, 0}});
  NodesMatch(
      &IsInsideDiamond, E, Z,
      {{-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}});
  NodesMatch(
      &IsInsideDiamond, H, X,
      {{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}, {2, 0}});
  NodesMatch(
      &IsInsideDiamond, H, Y,
      {{-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}});
  NodesMatch(
      &IsInsideDiamond, H, Z,
      {{-2, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, 0}});
  NodesMatch(
      &IsInsideDiamond, C, X,
      {{-2, 0}, {-2, 1}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}, {0, 0}, {0, 1}});
  NodesMatch(
      &IsInsideDiamond, C, Y,
      {{-2, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, 0}});
  NodesMatch(
      &IsInsideDiamond, C, Z,
      {{-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}});
}

TEST(Diamond, IsTrailingEdge) {
  NodesMatch(&IsTrailingEdge, E, X, {{-1, -1}, {-2, 0}, {-2, 1}, {-1, 2}});
  NodesMatch(&IsTrailingEdge, E, Y, {{-1, -1}, {-2, 0}, {-1, 1}});
  NodesMatch(&IsTrailingEdge, E, Z, {{0, -1}, {-1, 0}, {-1, 1}, {0, 2}});
  NodesMatch(&IsTrailingEdge, H, X, {{0, -1}, {-1, 0}, {0, 1}});
  NodesMatch(&IsTrailingEdge, H, Y, {{0, -1}, {-1, 0}, {-1, 1}, {0, 2}});
  NodesMatch(&IsTrailingEdge, H, Z, {{-1, -1}, {-2, 0}, {-1, 1}});
  NodesMatch(&IsTrailingEdge, C, X, {{-1, -1}, {-2, 0}, {-2, 1}, {-1, 2}});
  NodesMatch(&IsTrailingEdge, C, Y, {{-1, -1}, {-2, 0}, {-1, 1}});
  NodesMatch(&IsTrailingEdge, C, Z, {{0, -1}, {-1, 0}, {-1, 1}, {0, 2}});
}

TEST(Diamond, IsLeadingEdge) {
  NodesMatch(&IsLeadingEdge, E, X, {{0, -1}, {1, 0}, {1, 1}, {0, 2}});
  NodesMatch(&IsLeadingEdge, E, Y, {{1, -1}, {2, 0}, {1, 1}});
  NodesMatch(&IsLeadingEdge, E, Z, {{1, -1}, {2, 0}, {2, 1}, {1, 2}});
  NodesMatch(&IsLeadingEdge, H, X, {{1, -2}, {2, -1}, {3, 0}, {2, 1}, {1, 2}});
  NodesMatch(&IsLeadingEdge, H, Y, {{1, -1}, {2, 0}, {2, 1}, {1, 2}});
  NodesMatch(&IsLeadingEdge, H, Z, {{0, -2}, {1, -1}, {2, 0}, {1, 1}, {0, 2}});
  NodesMatch(&IsLeadingEdge, C, X, {{-1, -1}, {0, 0}, {0, 1}, {-1, 2}});
  NodesMatch(&IsLeadingEdge, C, Y, {{0, -1}, {1, 0}, {0, 1}});
  NodesMatch(&IsLeadingEdge, C, Z, {{0, -1}, {1, 0}, {1, 1}, {0, 2}});
}

TEST(Diamond, IsTopBotEy) { NodesMatch(&IsTopBotEy, E, Y, {{0, -2}, {0, 2}}); }

TEST(Diamond, IsActive) {
  NodesMatch(&IsActive, E, X,
             {{-2, 0},
              {-2, 1},
              {-1, -1},
              {-1, 0},
              {-1, 1},
              {-1, 2},
              {0, 0},
              {0, 1},
              {0, -1},
              {1, 0},
              {1, 1},
              {0, 2}});
  NodesMatch(&IsActive, E, Y,
             {{-2, 0},
              {-1, -1},
              {-1, 0},
              {-1, 1},
              {0, -1},
              {0, 0},
              {0, 1},
              {1, 0},
              {1, -1},
              {2, 0},
              {1, 1},
              {0, -2},
              {0, 2}});
  NodesMatch(&IsActive, E, Z,
             {{-1, 0},
              {-1, 1},
              {0, -1},
              {0, 0},
              {0, 1},
              {0, 2},
              {1, 0},
              {1, 1},
              {1, -1},
              {2, 0},
              {2, 1},
              {1, 2}});
  NodesMatch(&IsActive, H, X,
             {{-1, 0},
              {0, -1},
              {0, 0},
              {0, 1},
              {1, -1},
              {1, 0},
              {1, 1},
              {2, 0},
              {1, -2},
              {2, -1},
              {3, 0},
              {2, 1},
              {1, 2}});
  NodesMatch(&IsActive, H, Y,
             {{-1, 0},
              {-1, 1},
              {0, -1},
              {0, 0},
              {0, 1},
              {0, 2},
              {1, 0},
              {1, 1},
              {1, -1},
              {2, 0},
              {2, 1},
              {1, 2}});
  NodesMatch(&IsActive, H, Z,
             {{-2, 0},
              {-1, -1},
              {-1, 0},
              {-1, 1},
              {0, -1},
              {0, 0},
              {0, 1},
              {1, 0},
              {0, -2},
              {1, -1},
              {2, 0},
              {1, 1},
              {0, 2}});
  NodesMatch(
      &IsActive, C, X,
      {{-2, 0}, {-2, 1}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}, {0, 0}, {0, 1}});
  NodesMatch(
      &IsActive, C, Y,
      {{-2, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, 0}});
  NodesMatch(
      &IsActive, C, Z,
      {{-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}});
}

int NodeHash(Node n) {
  return n.i + (N / 2) +
         10 * (n.j + (N / 2) +
               10 * (n.k + 10 * (Index(n.ehc) + 10 * Index(n.xyz))));
}

TEST(Cell, Shift) {
  Cell<int> cell;
  for (Node n : AllNodes)
    if (IsActive(n))
      cell.Set(NodeHash(n), n);

  Shift(cell, E);
  Shift(cell, H);
  Shift(cell, C);

  for (Node n : AllNodes)
    if (IsActive(n) && IsActive(n.dI(+1)))
      EXPECT_EQ(cell.Get(n), NodeHash(n.dI(+1)));
}

} // namespace
} // namespace diamond
