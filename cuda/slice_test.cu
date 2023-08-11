#include <gtest/gtest.h>

#include "diamond.h"
#include "slice.h"
#include "testutils.h"

namespace slice {
namespace {

using defs::IsAux;
using diamond::X;
using diamond::Y;
using diamond::Z;

template <typename T> struct LayerKernelArgs {
  T *ptr;
  T *result;
  XY pos, domain;
};

template <typename T> __global__ void LayerKernel(LayerKernelArgs<T> args) {
  int t = defs::ThreadPos();

  Layer<T> layer;
  layer.Load(args.ptr, args.pos, args.domain, t);
  for (int i = 0; i < kWarpSize; ++i)
    args.result[t + kWarpSize * i] = layer.Get(i);
}

int LayerNodeHash(int index, XY pos) {
  return index + 100 * (pos.y + 100 * pos.x);
}

void TestLayer(XY domain) {
  int n = Layer<int>::GlobalElems(domain);
  testutils::Array<int> arr(n);
  for (int x = 0; x < domain.x; ++x)
    for (int y = 0; y < domain.y; y += 2)
      for (int index = 0; index < kWarpSize; ++index) {
        XY pos(x, y);
        arr[Layer<int>::GlobalIndex(index, pos, domain)] =
            LayerNodeHash(index, pos);
      }

  for (int x = 0; x < domain.x; ++x)
    for (int y = 0; y < domain.y; y += 2) {
      XY pos(x, y);
      // XY pos(3, 2);
      testutils::Array<int> out(kWarpSize * kWarpSize);
      LayerKernelArgs<int> args = {arr.Ptr(), out.Ptr(), pos, domain};
      testutils::LaunchCooperativeKernel((void *)&LayerKernel<int>,
                                         (void *)&args, UV(1, 1), UV(1, 1));
      for (int i = 0; i < kWarpSize; ++i)
        for (int j = 0; j < kWarpSize; ++j)
          EXPECT_EQ(out[j + kWarpSize * i], LayerNodeHash(i, pos));
    }
}

TEST(Slice, Layer) { TestLayer(/*domain=*/XY(4, 4)); }

// Only returns `+2`, `0`, `-4`, or `-6` because of how
// `ZMask::ConvertToCoeff()` works.
int ZMaskNodeHash(XY pos, Xyz xyz) {
  int hash = diamond::Index(xyz) + 100 * (pos.y + 100 * pos.x);
  hash = 2 * ((hash * 37) % 4);
  return hash < 4 ? hash : -hash;
}

void TestZMask(XY domain) {
  int dt = 1;
  testutils::Array<int> externalarr(ZMask<int>::ExternalElems(domain));
  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; ++j)
      for (Xyz xyz : diamond::AllXyz) {
        XY pos(i, j);
        externalarr[ZMask<int>::ExternalIndex(pos, xyz, domain)] =
            ZMaskNodeHash(pos, xyz);
      }

  testutils::Array<int> globalarr(ZMask<int>::GlobalElems(domain));
  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; j += 2)
      ZMask<int>::WriteGlobal(externalarr.Ptr(), globalarr.Ptr(), XY(i, j),
                              domain, dt);

  for (int x = 0; x < domain.x; ++x)
    for (int y = 0; y < domain.y; y += 2) {
      XY pos(x, y);
      int cnt = 0;
      for (int i : diamond::AllI)
#pragma unroll
        for (int j : diamond::AllJ)
#pragma unroll
          for (diamond::Xyz xyz : diamond::AllXyz) {
            Node n(i, j, /*k=*/0, E, xyz);
            if (diamond::IsInsideDiamond(
                    diamond::Node(i, j, /*k=*/0, E, xyz))) {
              XY p = pos + XY(i, j);
              bool isinside =
                  p.x >= 0 && p.y >= 0 && p.x < domain.x && p.y < domain.y;
              EXPECT_EQ(globalarr[ZMask<int>::GlobalIndex(cnt, pos, domain)],
                        isinside ? ZMask<int>::ConvertToCoeff(
                                       ZMaskNodeHash(p, xyz), dt)
                                 : 0)
                  << "(pos, node) = (" << pos << ", " << n << ")\n";
              ++cnt;
            }
          }
    }
}

TEST(Slice, ZMask) { TestZMask(/*domain=*/XY(6, 7)); }

int ZSrcNodeHash(XY pos, Xyz xyz, int channel) {
  return diamond::Index(xyz) + 100 * (pos.y + 100 * (pos.x + 100 * channel));
}

void TestZSrc(XY domain) {
  testutils::Array<int> externalarr(ZSrc<int>::ExternalElems(domain));
  for (int channel = 0; channel < 2; ++channel)
    for (int i = 0; i < domain.x; ++i)
      for (int j = 0; j < domain.y; ++j)
        for (Xyz xyz : {X, Y}) {
          XY pos(i, j);
          externalarr[ZSrc<int>::ExternalIndex(pos, xyz, channel, domain)] =
              ZSrcNodeHash(pos, xyz, channel);
        }

  testutils::Array<int> globalarr(ZSrc<int>::GlobalElems(domain));
  for (int i = 0; i < domain.x; ++i)
    for (int j = 0; j < domain.y; j += 2)
      ZSrc<int>::WriteGlobal(externalarr.Ptr(), globalarr.Ptr(), /*srcpos=*/0,
                             XY(i, j), domain);

  for (int x = 0; x < domain.x; ++x)
    for (int y = 0; y < domain.y; y += 2) {
      XY pos(x, y);
      int cnt = 0;
      for (int i : diamond::AllI)
#pragma unroll
        for (int j : diamond::AllJ)
#pragma unroll
          for (diamond::Xyz xyz : {X, Y}) {
            Node n(i, j, /*k=*/0, E, xyz);
            if (diamond::IsInsideDiamond(
                    diamond::Node(i, j, /*k=*/0, E, xyz))) {
              XY p = pos + XY(i, j);
              bool isinside =
                  p.x >= 0 && p.y >= 0 && p.x < domain.x && p.y < domain.y;
              // std::cout
              //     << globalarr[ZSrc<int>::GlobalIndex(cnt, XY(x, y),
              //     domain)]
              //     << " at (pos, node) = (" << XY(x, y) << ", "
              //     << Node(i, j, 0, E, xyz) << ")\n";
              for (int channel = 0; channel < 2; ++channel) {
                EXPECT_EQ(globalarr[ZSrc<int>::GlobalIndex(cnt, pos, domain)],
                          isinside ? ZSrcNodeHash(p, xyz, channel) : 0)
                    << "(pos, node, channel) = (" << pos << ", " << n << ", "
                    << channel << ")\n";
                ++cnt;
              }
            }
          }
    }
}

TEST(Slice, ZSrc) { TestZSrc(/*domain=*/XY(6, 7)); }

// TEST(Slice, YSrcGetHalf2) {
//   YSrc<half2> ysrc;
//
//   ((half2 *)ysrc.val)[4 * 1 + 3] = __floats2half2_rn(42.0f, 43.0f);
//   EXPECT_EQ(__low2float(ysrc.Get(/*i=*/1, /*k=*/1, /*xyz=*/Z)), 42.0f);
//   EXPECT_EQ(__high2float(ysrc.Get(/*i=*/1, /*k=*/1, /*xyz=*/Z)), 43.0f);
// }

int YSrcNodeHash(int x, int z, Xyz xyz) {
  return diamond::Index(xyz) + 100 * (z + 100 * x);
}

void TestYSrcWriteGlobal(int xx, int npml, int zshift) {
  testutils::Array<int> externalarr(YSrc<int>::ExternalElems(xx, npml));
  for (int x = 0; x < xx; ++x)
    for (int z = 0; z < ExtZz<int>(npml); ++z)
      for (Xyz xyz : {X, Z})
        externalarr[YSrc<int>::ExternalIndex(x, z, xyz, npml)] =
            YSrcNodeHash(x, z, xyz);

  testutils::Array<int> globalarr(YSrc<int>::GlobalElems(xx));
  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int x = 0; x < xx; ++x)
      YSrc<int>::WriteGlobal(externalarr.Ptr(), globalarr.Ptr(), x, threadpos,
                             xx, npml, zshift, IsAux(threadpos, npml));

  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int x = 0; x < xx; ++x)
      for (int k = 0; k < Nz; ++k)
        for (Xyz xyz : {X, Z})
          EXPECT_EQ(
              globalarr[YSrc<int>::GlobalIndex(x, k, xyz, threadpos, xx)],
              IsAux(threadpos, npml)
                  ? 0
                  : YSrcNodeHash(x, ExtZIndex<int>(k, threadpos, npml, zshift),
                                 xyz));
}

TEST(Slice, YSrcWriteGlobal) {
  TestYSrcWriteGlobal(/*xx=*/3, /*npml=*/2, /*zshift=*/1);
}

float Half2YSrcNodeHash(int x, int z, Xyz xyz) {
  return float(diamond::Index(xyz) + 10 * (z + 10 * x));
}

void TestHalf2YSrcWriteGlobal(int xx, int npml, int zshift) {
  testutils::Array<float> externalarr(YSrc<half2>::ExternalElems(xx, npml));
  for (int x = 0; x < xx; ++x)
    for (int z = 0; z < ExtZz<half2>(npml); ++z)
      for (Xyz xyz : {X, Z})
        externalarr[YSrc<half2>::ExternalIndex(x, z, xyz, npml)] =
            Half2YSrcNodeHash(x, z, xyz);

  testutils::Array<half2> globalarr(YSrc<half2>::GlobalElems(xx));
  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int x = 0; x < xx; ++x)
      YSrc<half2>::WriteGlobal(externalarr.Ptr(), globalarr.Ptr(), x, threadpos,
                               xx, npml, zshift, IsAux(threadpos, npml));

  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int x = 0; x < xx; ++x)
      for (int k = 0; k < Nz; ++k)
        for (Xyz xyz : {X, Z}) {
          EXPECT_EQ(
              __low2float(globalarr[YSrc<half2>::GlobalIndex(x, k, xyz,
                                                             threadpos, xx)]),
              IsAux(threadpos, npml)
                  ? 0
                  : Half2YSrcNodeHash(
                        x, ExtZIndex<half2>(k, threadpos, npml, zshift), xyz));
          EXPECT_EQ(
              __high2float(globalarr[YSrc<half2>::GlobalIndex(x, k, xyz,
                                                              threadpos, xx)]),
              IsAux(threadpos, npml)
                  ? 0
                  : Half2YSrcNodeHash(
                        x, ExtZIndex<half2>(k + Nz, threadpos, npml, zshift),
                        xyz));
        }
}

TEST(Slice, Half2YSrcWriteGlobal) {
  TestHalf2YSrcWriteGlobal(/*xx=*/3, /*npml=*/2, /*zshift=*/1);
}

void TestYSrc(int xx) {
  int n = YSrc<int>::GlobalElems(xx);
  testutils::Array<int> arr(n);

  // Fill.
  for (int threadpos = 0; threadpos < kWarpSize; ++threadpos)
    for (int x = 0; x < xx; ++x)
      for (int k = 0; k < Nz; ++k)
        for (Xyz xyz : {X, Z})
          arr[YSrc<int>::GlobalIndex(x, k, xyz, threadpos, xx)] =
              YSrcNodeHash(x, k + Nz * threadpos, xyz);

  YSrc<int> line;
  int threadpos = 23; // TODO: Check all.
  for (int x = 0; x < xx; ++x) {
    line.Load(arr.Ptr(), x, threadpos);
    if (x >= 2 && x < xx - 1)
      for (int i = 0; i < 3; ++i)
        for (int k = 0; k < Nz; ++k) {
          EXPECT_EQ(YSrcNodeHash(x - i, k + Nz * threadpos, X),
                    line.Get(i, k, X))
              << "(x, i, k, xyz, threadpos) = (" << x << ", " << i << ", " << k
              << ", " << X << ", " << threadpos << ")\n";
          EXPECT_EQ(YSrcNodeHash(x - i + 1, k + Nz * threadpos, Z),
                    line.Get(i, k, Z))
              << "(x, i, k, xyz, threadpos) = (" << x << ", " << i << ", " << k
              << ", " << Z << ", " << threadpos << ")\n";
        }
  }
}

TEST(Slice, YSrc) { TestYSrc(/*xx=*/5); }

// TODO: For all of these, test with zshift and half2 conversion.

// TODO: Test `Out` by filling up global memory with a simple hash (base 10)
// and then checking that writes at every eligible position produce the
// correct results.

// TODO: Test `Profile` by filling up global memory with a simple hash and
// checking loaded values.

// TODO: Test `Layer` by filling up a layer, running the kernel to convert it
// into the format that we like, and then loading it at every pos and checking
// that we have the correct values.

// TODO: Test `YSrc` by filling up global memory, and checking against pos in
// a raster-scan like manner.

} // namespace
} // namespace slice
