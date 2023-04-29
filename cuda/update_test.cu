#include <gtest/gtest.h>

#include "testutils.h"
#include "update.h"

namespace update {
namespace {

using defs::kWarpSize;
using defs::One;
using defs::Zero;
using diamond::kNumEhc;
using diamond::kNumXyz;
using diamond::N;
using diamond::Nz;

constexpr const int kWarpCellElems =
    (N + 2) * (N + 1) * Nz * kWarpSize * kNumEhc * kNumXyz;

__dh__ int WarpCellIndex(Node n, int threadpos) {
  return (n.i + 2) +
         (N + 2) * ((n.j + 2) +
                    (N + 1) * (n.k + Nz * (diamond::Index(n.ehc) +
                                           kNumEhc * (diamond::Index(n.xyz) +
                                                      kNumXyz * threadpos))));
}

struct ShuffleKernelArgs {
  int *circout, *swapout;
  int offset, p;
};

__global__ void ShuffleKernel(ShuffleKernelArgs args) {
  int t = defs::ThreadPos();
  args.circout[t] = CircShuffle(t, args.offset, args.p);
  args.swapout[t] = SwapShuffle(t);
}

void RunShuffleKernel(int offset, int p) {
  testutils::Array<int> circout(kWarpSize), swapout(kWarpSize);

  ShuffleKernelArgs args = {circout.Ptr(), swapout.Ptr(), offset, p};
  testutils::LaunchCooperativeKernel((void *)ShuffleKernel, (void *)&args,
                                     /*blockshape=**/ UV(1, 1),
                                     /*gridshape=*/UV(1, 1));

  // Run test.
  for (int i = 0; i < kWarpSize; ++i) {
    std::cout << i << ": " << circout[i] << " " << swapout[i] << "\n";
  }
}

TEST(Update, Shuffle) { RunShuffleKernel(+1, 6); }

template <typename T> struct CurlKernelArgs {
  T *ptr, *a, *b, d;
  int p;
  Ehc ehc;
};

template <typename T> __global__ void CurlKernel(CurlKernelArgs<T> args) {
  int t = defs::ThreadPos();

  // Load cell.
  Cell<T> cell;
  for (Node n : diamond::AllNodes)
    cell.Set(args.ptr[WarpCellIndex(n, t)], n);

  // Load constants.
  const T a[Nz] = {args.a[Nz * t], args.a[Nz * t + 1]};
  const T b[Nz] = {args.b[Nz * t], args.b[Nz * t + 1]};

  ZCoeff<T> zcoeff;
  for (int k = 0; k < Nz; ++k) {
    zcoeff.Set(a[k], k, args.ehc, zcoeff::InternalType::A);
    zcoeff.Set(b[k], k, args.ehc, zcoeff::InternalType::B);
  }

  // Perform update.
  Update(cell, zcoeff, args.d, args.p, defs::IsAux(t, args.p), args.ehc);

  // Write cell.
  for (Node n : diamond::AllNodes)
    args.ptr[WarpCellIndex(n, t)] = cell.Get(n);
}

template <typename T, int N> void Copy(std::array<T, N> src, T *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = src[i];
}

template <typename T, int N> void Copy(T *src, std::array<T, N> &dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = src[i];
}

template <typename T>
std::array<T, kWarpCellElems> RunCurlKernel(std::array<T, kWarpCellElems> arrwc,
                                            std::array<T, Nz * kWarpSize> arra,
                                            std::array<T, Nz * kWarpSize> arrb,
                                            T d, int p, Ehc ehc) {
  // Transfer to GPU-compatible arrays.
  testutils::Array<T> wc(kWarpCellElems), a(Nz * kWarpSize), b(Nz * kWarpSize);

  Copy<int, kWarpCellElems>(arrwc, wc.Ptr());
  Copy<int, Nz * kWarpSize>(arra, a.Ptr());
  Copy<int, Nz * kWarpSize>(arrb, b.Ptr());

  CurlKernelArgs<T> args = {wc.Ptr(), a.Ptr(), b.Ptr(), d, p, ehc};
  testutils::LaunchCooperativeKernel((void *)CurlKernel<T>, (void *)&args,
                                     /*blockshape=**/ UV(1, 1),
                                     /*gridshape=*/UV(1, 1));

  // Transfer to output array.
  std::array<T, kWarpCellElems> output = {};
  Copy<int, kWarpCellElems>(wc.Ptr(), output);
  return output;
}

template <typename T> struct NTV {
  NTV(Node n, int t, T val) : n(n), t(t), val(val) {}

  Node n;
  int t;
  T val;
};

template <typename T> struct ITV {
  ITV(int index, int t, T val) : index(index), t(t), val(val) {}

  int index;
  int t;
  T val;
};

template <typename T>
void UpdateNodeTest(T d, int p, Ehc ehc,       //
                    std::vector<ITV<T>> inita, //
                    std::vector<ITV<T>> initb, //
                    std::vector<NTV<T>> init,  //
                    NTV<T> expected) {
  std::array<int, kWarpCellElems> wc = {};
  std::array<int, Nz *kWarpSize> arra = {};
  std::array<int, Nz *kWarpSize> arrb = {};

  for (auto iv : inita)
    arra[iv.index + Nz * iv.t] = iv.val;

  for (auto iv : initb)
    arrb[iv.index + Nz * iv.t] = iv.val;

  for (auto ntv : init)
    wc[WarpCellIndex(ntv.n, ntv.t)] = ntv.val;

  auto output = RunCurlKernel(wc, arra, arrb, d, p, ehc);

  EXPECT_EQ(output[WarpCellIndex(expected.n, expected.t)], expected.val)
      << "at node " << expected.n;
}

TEST(Update, NoPml) {
  int t = 15;

  UpdateNodeTest(/*d=*/1, /*p=*/0, /*ehc=*/E,
                 /*inita=*/{ITV<int>(0, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, X), t, 2),
                     NTV<int>(Node(0, 0, 0, C, X), t, 10),
                     NTV<int>(Node(1, 0, 0, H, Z), t, 7),
                     NTV<int>(Node(1, -1, 0, H, Z), t, -50),
                     NTV<int>(Node(1, 0, 0, H, Y), t, -4),
                     NTV<int>(Node(1, 0, 1, H, Y), t - 1, 80),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, X), t, 84572));

  UpdateNodeTest(/*d=*/1, /*p=*/0, /*ehc=*/E,
                 /*inita=*/{ITV<int>(1, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, X), t, 2),
                     NTV<int>(Node(0, 0, 1, C, X), t, 10),
                     NTV<int>(Node(1, 0, 1, H, Z), t, 7),
                     NTV<int>(Node(1, -1, 1, H, Z), t, -50),
                     NTV<int>(Node(1, 0, 1, H, Y), t, -4),
                     NTV<int>(Node(1, 0, 0, H, Y), t, 80),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, X), t, 84572));

  UpdateNodeTest(/*d=*/1, /*p=*/0, /*ehc=*/E,
                 /*inita=*/{ITV<int>(0, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, Y), t, 2),
                     NTV<int>(Node(0, 0, 0, C, Y), t, 1),
                     NTV<int>(Node(1, 0, 0, H, X), t, 35),
                     NTV<int>(Node(1, 0, 1, H, X), t - 1, -250),
                     NTV<int>(Node(1, 0, 0, H, Z), t, -4000),
                     NTV<int>(Node(0, 0, 0, H, Z), t, 80000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, Y), t, 84572));

  UpdateNodeTest(/*d=*/1, /*p=*/0, /*ehc=*/E,
                 /*inita=*/{ITV<int>(1, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, Y), t, 2),
                     NTV<int>(Node(0, 0, 1, C, Y), t, 1),
                     NTV<int>(Node(1, 0, 1, H, X), t, 35),
                     NTV<int>(Node(1, 0, 0, H, X), t, -250),
                     NTV<int>(Node(1, 0, 1, H, Z), t, -4000),
                     NTV<int>(Node(0, 0, 1, H, Z), t, 80000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, Y), t, 84572));

  UpdateNodeTest(/*d=*/1, /*p=*/0, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, Z), t, 2),
                     NTV<int>(Node(0, 0, 0, C, Z), t, 10),
                     NTV<int>(Node(1, 0, 0, H, Y), t, 7),
                     NTV<int>(Node(0, 0, 0, H, Y), t, -50),
                     NTV<int>(Node(1, 0, 0, H, X), t, -400),
                     NTV<int>(Node(1, -1, 0, H, X), t, 8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, Z), t, 84572));

  UpdateNodeTest(/*d=*/1, /*p=*/0, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, Z), t, 2),
                     NTV<int>(Node(0, 0, 1, C, Z), t, 10),
                     NTV<int>(Node(1, 0, 1, H, Y), t, 7),
                     NTV<int>(Node(0, 0, 1, H, Y), t, -50),
                     NTV<int>(Node(1, 0, 1, H, X), t, -400),
                     NTV<int>(Node(1, -1, 1, H, X), t, 8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, Z), t, 84572));

  UpdateNodeTest(/*d=*/-10, /*p=*/0, /*ehc=*/H,
                 /*inita=*/{ITV<int>(0, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, X), t, 2),
                     NTV<int>(Node(0, 1, 0, E, Z), t, -7),
                     NTV<int>(Node(0, 0, 0, E, Z), t, 50),
                     NTV<int>(Node(0, 0, 1, E, Y), t, 4),
                     NTV<int>(Node(0, 0, 0, E, Y), t, -80),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, X), t, 84572));

  UpdateNodeTest(/*d=*/-10, /*p=*/0, /*ehc=*/H,
                 /*inita=*/{ITV<int>(1, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, X), t, 2),
                     NTV<int>(Node(0, 1, 1, E, Z), t, -7),
                     NTV<int>(Node(0, 0, 1, E, Z), t, 50),
                     NTV<int>(Node(0, 0, 0, E, Y), t + 1, 4),
                     NTV<int>(Node(0, 0, 1, E, Y), t, -80),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, X), t, 84572));

  UpdateNodeTest(/*d=*/-1, /*p=*/0, /*ehc=*/H,
                 /*inita=*/{ITV<int>(0, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, Y), t, 2),
                     NTV<int>(Node(0, 0, 1, E, X), t, -35),
                     NTV<int>(Node(0, 0, 0, E, X), t, 250),
                     NTV<int>(Node(1, 0, 0, E, Z), t, 4000),
                     NTV<int>(Node(0, 0, 0, E, Z), t, -80000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, Y), t, 84572));

  UpdateNodeTest(/*d=*/-1, /*p=*/0, /*ehc=*/H,
                 /*inita=*/{ITV<int>(1, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, Y), t, 2),
                     NTV<int>(Node(0, 0, 0, E, X), t + 1, -35),
                     NTV<int>(Node(0, 0, 1, E, X), t, 250),
                     NTV<int>(Node(1, 0, 1, E, Z), t, 4000),
                     NTV<int>(Node(0, 0, 1, E, Z), t, -80000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, Y), t, 84572));

  UpdateNodeTest(/*d=*/-10, /*p=*/0, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, Z), t, 2),
                     NTV<int>(Node(1, 0, 0, E, Y), t, -7),
                     NTV<int>(Node(0, 0, 0, E, Y), t, 50),
                     NTV<int>(Node(0, 1, 0, E, X), t, 400),
                     NTV<int>(Node(0, 0, 0, E, X), t, -8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, Z), t, 84572));

  UpdateNodeTest(/*d=*/-10, /*p=*/0, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, Z), t, 2),
                     NTV<int>(Node(1, 0, 1, E, Y), t, -7),
                     NTV<int>(Node(0, 0, 1, E, Y), t, 50),
                     NTV<int>(Node(0, 1, 1, E, X), t, 400),
                     NTV<int>(Node(0, 0, 1, E, X), t, -8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, Z), t, 84572));
}

TEST(Update, PmlNormalThreadInside) {
  int t = 5;
  int s = kWarpSize - (t + 1);

  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{ITV<int>(0, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, X), t, 2),
                     NTV<int>(Node(0, 0, 0, C, X), t, 10),
                     NTV<int>(Node(1, 0, 0, H, Z), t, 7),
                     NTV<int>(Node(1, -1, 0, H, Z), t, -50),
                     NTV<int>(Node(1, 0, 0, H, Y), t, -4),
                     NTV<int>(Node(1, 0, 1, H, Y), t - 1, 80),
                     NTV<int>(Node(0, 0, 0, E, X), s, 30000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, X), t, 384572));

  UpdateNodeTest(/*d=*/1, /*p=*/8, /*ehc=*/E,
                 /*inita=*/{ITV<int>(1, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, X), t, 2),
                     NTV<int>(Node(0, 0, 1, C, X), t, 10),
                     NTV<int>(Node(1, 0, 1, H, Z), t, 7),
                     NTV<int>(Node(1, -1, 1, H, Z), t, -50),
                     NTV<int>(Node(1, 0, 1, H, Y), t, -4),
                     NTV<int>(Node(1, 0, 0, H, Y), t, 80),
                     NTV<int>(Node(0, 0, 1, E, X), s, 30000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, X), t, 384572));

  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{ITV<int>(0, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, Y), t, 2),
                     NTV<int>(Node(0, 0, 0, C, Y), t, 1),
                     NTV<int>(Node(1, 0, 0, H, X), t, 35),
                     NTV<int>(Node(1, 0, 1, H, X), t - 1, -250),
                     NTV<int>(Node(1, 0, 0, H, Z), t, -4000),
                     NTV<int>(Node(0, 0, 0, H, Z), t, 80000),
                     NTV<int>(Node(0, 0, 0, E, Y), s, 300000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, Y), t, 384572));

  UpdateNodeTest(/*d=*/1, /*p=*/8, /*ehc=*/E,
                 /*inita=*/{ITV<int>(1, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, Y), t, 2),
                     NTV<int>(Node(0, 0, 1, C, Y), t, 1),
                     NTV<int>(Node(1, 0, 1, H, X), t, 35),
                     NTV<int>(Node(1, 0, 0, H, X), t, -250),
                     NTV<int>(Node(1, 0, 1, H, Z), t, -4000),
                     NTV<int>(Node(0, 0, 1, H, Z), t, 80000),
                     NTV<int>(Node(0, 0, 1, E, Y), s, 300000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, Y), t, 384572));

  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, Z), t, 2),
                     NTV<int>(Node(0, 0, 0, C, Z), t, 10),
                     NTV<int>(Node(1, 0, 0, H, Y), t, 7),
                     NTV<int>(Node(0, 0, 0, H, Y), t, -50),
                     NTV<int>(Node(1, 0, 0, H, X), t, -400),
                     NTV<int>(Node(1, -1, 0, H, X), t, 8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, Z), t, 84572));

  UpdateNodeTest(/*d=*/1, /*p=*/8, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, Z), t, 2),
                     NTV<int>(Node(0, 0, 1, C, Z), t, 10),
                     NTV<int>(Node(1, 0, 1, H, Y), t, 7),
                     NTV<int>(Node(0, 0, 1, H, Y), t, -50),
                     NTV<int>(Node(1, 0, 1, H, X), t, -400),
                     NTV<int>(Node(1, -1, 1, H, X), t, 8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, Z), t, 84572));

  UpdateNodeTest(/*d=*/-10, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{ITV<int>(0, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, X), t, 2),
                     NTV<int>(Node(0, 1, 0, E, Z), t, -7),
                     NTV<int>(Node(0, 0, 0, E, Z), t, 50),
                     NTV<int>(Node(0, 0, 1, E, Y), t, 4),
                     NTV<int>(Node(0, 0, 0, E, Y), t, -80),
                     NTV<int>(Node(0, 0, 0, H, X), s, -30000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, X), t, 384572));

  UpdateNodeTest(/*d=*/-10, /*p=*/8, /*ehc=*/H,
                 /*inita=*/{ITV<int>(1, t, 100)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, X), t, 2),
                     NTV<int>(Node(0, 1, 1, E, Z), t, -7),
                     NTV<int>(Node(0, 0, 1, E, Z), t, 50),
                     NTV<int>(Node(0, 0, 0, E, Y), t + 1, 4),
                     NTV<int>(Node(0, 0, 1, E, Y), t, -80),
                     NTV<int>(Node(0, 0, 1, H, X), s, -30000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, X), t, 384572));

  UpdateNodeTest(/*d=*/-1, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{ITV<int>(0, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, Y), t, 2),
                     NTV<int>(Node(0, 0, 1, E, X), t, -35),
                     NTV<int>(Node(0, 0, 0, E, X), t, 250),
                     NTV<int>(Node(1, 0, 0, E, Z), t, 4000),
                     NTV<int>(Node(0, 0, 0, E, Z), t, -80000),
                     NTV<int>(Node(0, 0, 0, H, Y), s, -300000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, Y), t, 384572));

  UpdateNodeTest(/*d=*/-1, /*p=*/8, /*ehc=*/H,
                 /*inita=*/{ITV<int>(1, t, 2)},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, Y), t, 2),
                     NTV<int>(Node(0, 0, 0, E, X), t + 1, -35),
                     NTV<int>(Node(0, 0, 1, E, X), t, 250),
                     NTV<int>(Node(1, 0, 1, E, Z), t, 4000),
                     NTV<int>(Node(0, 0, 1, E, Z), t, -80000),
                     NTV<int>(Node(0, 0, 1, H, Y), s, -300000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, Y), t, 384572));

  UpdateNodeTest(/*d=*/-10, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, Z), t, 2),
                     NTV<int>(Node(1, 0, 0, E, Y), t, -7),
                     NTV<int>(Node(0, 0, 0, E, Y), t, 50),
                     NTV<int>(Node(0, 1, 0, E, X), t, 400),
                     NTV<int>(Node(0, 0, 0, E, X), t, -8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, Z), t, 84572));

  UpdateNodeTest(/*d=*/-10, /*p=*/8, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, Z), t, 2),
                     NTV<int>(Node(1, 0, 1, E, Y), t, -7),
                     NTV<int>(Node(0, 0, 1, E, Y), t, 50),
                     NTV<int>(Node(0, 1, 1, E, X), t, 400),
                     NTV<int>(Node(0, 0, 1, E, X), t, -8000),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, Z), t, 84572));
}

TEST(Update, PmlAuxThread) {
  int t = 29;
  int s = kWarpSize - (t + 1);

  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(0, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, X), t, 2),
                     NTV<int>(Node(0, 0, 0, C, X), t, 10),
                     NTV<int>(Node(1, 0, 0, H, Y), s, -3),
                     NTV<int>(Node(1, 0, 1, H, Y), s - 1, 20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, X), t, 692));

  UpdateNodeTest(/*d=*/1, /*p=*/8, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(1, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, X), t, 2),
                     NTV<int>(Node(0, 0, 1, C, X), t, 10),
                     NTV<int>(Node(1, 0, 1, H, Y), s, -3),
                     NTV<int>(Node(1, 0, 0, H, Y), s, 20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, X), t, 692));

  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(0, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, E, Y), t, 2),
                     NTV<int>(Node(0, 0, 0, C, Y), t, 10),
                     NTV<int>(Node(1, 0, 0, H, X), s, 3),
                     NTV<int>(Node(1, 0, 1, H, X), s - 1, -20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, Y), t, 692));

  UpdateNodeTest(/*d=*/1, /*p=*/8, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(1, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, E, Y), t, 2),
                     NTV<int>(Node(0, 0, 1, C, Y), t, 10),
                     NTV<int>(Node(1, 0, 1, H, X), s, 3),
                     NTV<int>(Node(1, 0, 0, H, X), s, -20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, Y), t, 692));

  // Ez/Hz nodes should not be updated.
  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/{NTV<int>(Node(0, 0, 0, E, Z), t, 42)},
                 /*expected=*/NTV<int>(Node(0, 0, 0, E, Z), t, 42));

  UpdateNodeTest(/*d=*/1, /*p=*/7, /*ehc=*/E,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/{NTV<int>(Node(0, 0, 1, E, Z), t, 42)},
                 /*expected=*/NTV<int>(Node(0, 0, 1, E, Z), t, 42));

  UpdateNodeTest(/*d=*/-10, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(0, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, X), t, 2),
                     NTV<int>(Node(0, 0, 1, E, Y), s, 3),
                     NTV<int>(Node(0, 0, 0, E, Y), s, -20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, X), t, 692));

  UpdateNodeTest(/*d=*/-10, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(1, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, X), t, 2),
                     NTV<int>(Node(0, 0, 0, E, Y), s + 1, 3),
                     NTV<int>(Node(0, 0, 1, E, Y), s, -20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, X), t, 692));

  UpdateNodeTest(/*d=*/-10, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(0, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 0, H, Y), t, 2),
                     NTV<int>(Node(0, 0, 1, E, X), s, -3),
                     NTV<int>(Node(0, 0, 0, E, X), s, 20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, Y), t, 692));

  UpdateNodeTest(/*d=*/-10, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{ITV<int>(1, s, 3)},
                 /*init=*/
                 {
                     NTV<int>(Node(0, 0, 1, H, Y), t, 2),
                     NTV<int>(Node(0, 0, 0, E, X), s + 1, -3),
                     NTV<int>(Node(0, 0, 1, E, X), s, 20),
                 },
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, Y), t, 692));

  UpdateNodeTest(/*d=*/-1, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/{NTV<int>(Node(0, 0, 0, H, Z), t, 42)},
                 /*expected=*/NTV<int>(Node(0, 0, 0, H, Z), t, 42));

  UpdateNodeTest(/*d=*/-1, /*p=*/7, /*ehc=*/H,
                 /*inita=*/{},
                 /*initb=*/{},
                 /*init=*/{NTV<int>(Node(0, 0, 1, H, Z), t, 42)},
                 /*expected=*/NTV<int>(Node(0, 0, 1, H, Z), t, 42));
}

// template <typename T>
// void AddAbsTest(T absval, const T bval, T expected, bool isaux, RunShape rs)
// {
//   Cell<T> cell;
//   diamond::InitCell(cell, One<T>());
//
//   int nelems = layer::SharedElems(rs.nabs, rs);
//   testutils::Array<T> arr(nelems);
//
//   for (int i = 0; i < nelems; ++i)
//     arr[i] = absval;
//
//   T b[Nz];
//   for (int i = 0; i < Nz; ++i)
//     b[i] = bval;
//
//   // Warp position and step should not matter here since the layer is
//   uniformly
//   // of value `absval`.
//   AddAbs(cell, arr.Ptr(), b, rs.npml, isaux, /*warppos=*/UV(0, 0),
//   /*step=*/0,
//          rs, E);
//
//   for (Node n : diamond::AllNodes)
//     if (diamond::IsInsideDiamond(n) && n.ehc == E)
//       EXPECT_EQ(cell.Get(n), expected);
// }
//
// TEST(Update, AddAbs) {
//   RunShape rs(/*blockshape=*/UV(2, 4),
//               /*gridshape=*/UV(0, 0),
//               /*spacing=*/0,
//               /*domain=*/XY(0, 0),
//               /*srctype=*/RunShape::SourceType::DEBUG,
//               /*nout=*/0,
//               /*npml=*/5);
//   AddAbsTest(/*absval=*/2, /*bval=*/3, /*expected=*/2, /*isaux=*/false, rs);
//   AddAbsTest(/*absval=*/2, /*bval=*/3, /*expected=*/3, /*isaux=*/true, rs);
// }
//
// template <typename T>
// void AddSrcTest(T srcval, const T a[2], int srcpos, T expected, int k,
//                 RunShape rs) {
//   Cell<T> cell;
//   diamond::InitCell(cell, Zero<T>());
//
//   int nelems = layer::SharedElems(rs.nsrc, rs);
//   testutils::Array<T> arr(nelems);
//
//   for (int i = 0; i < nelems; ++i)
//     arr[i] = srcval;
//
//   AddSrc(cell, arr.Ptr(), a, srcpos, /*warppos=*/UV(0, 0), /*step=*/0, rs);
//
//   for (Node n : diamond::AllNodes)
//     if (diamond::IsInsideDiamond(n) && n.k == k && n.ehc == E && n.xyz != Z)
//       EXPECT_EQ(cell.Get(n), expected);
// }
//
// TEST(Update, AddSrc) {
//   RunShape rs(/*blockshape=*/UV(2, 4),
//               /*gridshape=*/UV(0, 0),
//               /*spacing=*/0,
//               /*domain=*/XY(0, 0),
//               /*srctype=*/RunShape::SourceType::ZSLICE,
//               /*nout=*/0,
//               /*npml=*/0);
//   int a[2] = {2, 3};
//   AddSrcTest(/*srcval=*/4, a, /*srcpos=*/16, /*expected=*/20, /*k=*/0, rs);
//   AddSrcTest(/*srcval=*/4, a, /*srcpos=*/16, /*expected=*/0, /*k=*/1, rs);
//   AddSrcTest(/*srcval=*/4, a, /*srcpos=*/17, /*expected=*/0, /*k=*/0, rs);
//   AddSrcTest(/*srcval=*/4, a, /*srcpos=*/17, /*expected=*/20, /*k=*/1, rs);
// }

// TODO: Restore.
// template <int N>
// void WriteOutputTest(Node n, XYT domainpos, int threadpos, int outsteps[N],
//                      XY domain, int outindex, bool iswrite) {
//   Cell<int> cell;
//   diamond::InitCell(cell, 0);
//   cell.Set(2001, n);
//
//   int numelems = field::NumElems(domain);
//   testutils::Array<int> arr(N * numelems);
//   for (int i = 0; i < N * numelems; ++i)
//     arr[i] = 42;
//
//   int *outputs[N];
//   for (int i = 0; i < N; ++i)
//     outputs[i] = arr.Ptr() + i * numelems;
//
//   WriteOutput<int, N>(cell, domainpos, threadpos, outsteps, outputs, domain);
//
//   int index = numelems * outindex +
//               field::Index(n.dI(domainpos.x).dJ(domainpos.y).dK(Nz *
//               threadpos),
//                            domain);
//   EXPECT_EQ(arr[index], (iswrite ? 2001 : 42)) << "domainpos = " <<
//   domainpos;
// }
//
// TEST(Update, WriteOutput) {
//   int outputsteps[3] = {11, 21, 31};
//   WriteOutputTest<3>(Node(0, 0, 0, E, X),
//                      /*domainpos=*/XYT(4, 9, 11),
//                      /*threadpos=*/14, //
//                      outputsteps,
//                      /*domain=*/XY(10, 13),
//                      /*outindex=*/0,
//                      /*iswrite=*/true);
//   WriteOutputTest<3>(Node(0, 0, 0, E, X),
//                      /*domainpos=*/XYT(3, 5, 21),
//                      /*threadpos=*/14, //
//                      outputsteps,
//                      /*domain=*/XY(10, 13),
//                      /*outindex=*/1,
//                      /*iswrite=*/true);
//   WriteOutputTest<3>(Node(0, 0, 0, E, X),
//                      /*domainpos=*/XYT(7, 3, 31),
//                      /*threadpos=*/14, //
//                      outputsteps,
//                      /*domain=*/XY(10, 13),
//                      /*outindex=*/2,
//                      /*iswrite=*/true);
//   WriteOutputTest<3>(Node(0, 0, 0, E, X),
//                      /*domainpos=*/XYT(7, 3, 12),
//                      /*threadpos=*/14, //
//                      outputsteps,
//                      /*domain=*/XY(10, 13),
//                      /*outindex=*/0,
//                      /*iswrite=*/false);
// }

} // namespace
} // namespace update
