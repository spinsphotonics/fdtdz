#include <gtest/gtest.h>

#include "diamond.h"
#include "testutils.h"
#include "zcoeff.h"

namespace zcoeff {
namespace {

using defs::IsAux;
using diamond::X;
using diamond::Y;
using diamond::Z;

TEST(ZCoeff, ZCoeff) {
  int zshift = 9;
  int npml = 5;
  testutils::Array<float> arr(ExternalElems<float>(npml));

  arr[ExternalIndex(50, H, ExternalType::A)] = 42.0f;
  arr[ExternalIndex(50, H, ExternalType::B)] = 43.0f;
  arr[ExternalIndex(50, H, ExternalType::Z)] = 44.0f;

  {
    ZCoeff<float> zcoeff;
    Load(zcoeff, arr.Ptr(), /*threadpos=*/2, npml, zshift, /*isaux=*/false);
    EXPECT_EQ(zcoeff.Get(/*k=*/1, H, InternalType::A), 42.0f + 44.0f);
    EXPECT_EQ(zcoeff.Get(/*k=*/1, H, InternalType::B), 42.0f);
  }

  {
    ZCoeff<float> zcoeff;
    Load(zcoeff, arr.Ptr(), /*threadpos=*/2, npml, zshift, /*isaux=*/true);
    EXPECT_EQ(zcoeff.Get(/*k=*/1, H, InternalType::A), 0.0f);
    EXPECT_EQ(zcoeff.Get(/*k=*/1, H, InternalType::B), 43.0f);
  }
}

TEST(ZCoeff, ZCoeffHalf2) {
  int zshift = 7;
  int npml = 5;
  testutils::Array<float> arr(ExternalElems<float>(npml));

  // Translates to low values.
  arr[ExternalIndex(106, H, ExternalType::A)] = 42.0f;
  arr[ExternalIndex(106, H, ExternalType::B)] = 43.0f;
  arr[ExternalIndex(106, H, ExternalType::Z)] = 44.0f;

  // Translates to high values.
  arr[ExternalIndex(0, H, ExternalType::A)] = 102.0f;
  arr[ExternalIndex(0, H, ExternalType::B)] = 103.0f;
  arr[ExternalIndex(0, H, ExternalType::Z)] = 104.0f;

  {
    ZCoeff<half2> zcoeff;
    Load(zcoeff, arr.Ptr(), /*threadpos=*/1, npml, zshift, /*isaux=*/false);
    EXPECT_EQ(__low2float(zcoeff.Get(/*k=*/1, H, InternalType::A)),
              42.0f + 44.0f);
    EXPECT_EQ(__low2float(zcoeff.Get(/*k=*/1, H, InternalType::B)), 42.0f);
    EXPECT_EQ(__high2float(zcoeff.Get(/*k=*/1, H, InternalType::A)),
              102.0f + 104.0f);
    EXPECT_EQ(__high2float(zcoeff.Get(/*k=*/1, H, InternalType::B)), 102.0f);
  }

  {
    ZCoeff<half2> zcoeff;
    Load(zcoeff, arr.Ptr(), /*threadpos=*/30, npml, zshift, /*isaux=*/true);
    EXPECT_EQ(__low2float(zcoeff.Get(/*k=*/1, H, InternalType::A)), 0.0f);
    EXPECT_EQ(__low2float(zcoeff.Get(/*k=*/1, H, InternalType::B)), 43.0f);
    EXPECT_EQ(__high2float(zcoeff.Get(/*k=*/1, H, InternalType::A)), 0.0f);
    EXPECT_EQ(__high2float(zcoeff.Get(/*k=*/1, H, InternalType::B)), 103.0f);
  }
}

} // namespace
} // namespace zcoeff
