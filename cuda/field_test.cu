#include <gtest/gtest.h>

#include "diamond.h"
#include "field.h"
#include "testutils.h"

namespace field {
namespace {

using defs::IsAux;
using diamond::X;
using diamond::Y;
using diamond::Z;

// Just a simple point test.
TEST(Field, Field) {
  XY domain(10, 10);
  int npml = 5;
  int zshift = 10;
  int nout = 4;

  testutils::Array<int> arr(ExternalElems<int>(domain, nout, npml));
  Cell<int> cell;
  InitCell(cell, 0);
  cell.Set(42, Node(0, 0, 1, diamond::E, diamond::X));
  XY pos(5, 5);
  int threadpos(3);
  int outindex = 2;
  WriteCell(cell, arr.Ptr(), pos, outindex, threadpos, domain, npml, zshift,
            /*isaux=*/false);
  EXPECT_EQ(arr[ExtNodeIndex(Node(5, 5, 51, diamond::E, diamond::X), outindex,
                             /*xx=*/10,
                             /*yy=*/10, ExtZz<int>(npml))],
            42);
}

TEST(Field, FieldHalf2) {
  XY domain(10, 10);
  int nout = 4;
  int npml = 5;
  int zshift = 7;
  testutils::Array<float> arr(ExternalElems<half2>(domain, nout, npml));
  Cell<half2> cell;
  InitCell(cell, defs::Zero<half2>());
  cell.Set(__floats2half2_rn(42.0f, 43.0f),
           Node(0, 0, 1, diamond::E, diamond::X));
  XY pos(5, 5);
  int threadpos(1);
  int outindex = 2;
  WriteCell(cell, arr.Ptr(), pos, outindex, threadpos, domain, npml, zshift,
            /*isaux=*/false);
  EXPECT_EQ(arr[ExtNodeIndex(Node(5, 5, 106, diamond::E, diamond::X), outindex,
                             /*xx=*/10, /*yy=*/10, ExtZz<half2>(npml))],
            42.0f);
  // Note that the lo-hi values of the half2 are on "opposite" sides of the
  // wrap-around.
  EXPECT_EQ(arr[ExtNodeIndex(Node(5, 5, 0, diamond::E, diamond::X), outindex,
                             /*xx=*/10, /*yy=*/10, ExtZz<half2>(npml))],
            43.0f);
}

} // namespace
} // namespace field
