// Defines the FDTD simulation kernel.

#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cooperative_groups.h>

#include "buffer.h"
#include "cbuf.h"
#include "defs.h"
#include "field.h"
#include "scanner.h"
#include "slice.h"
#include "update.h"
#include "zcoeff.h"

namespace kernel {

using defs::RunShape;
using defs::UV;
using defs::XY;
using defs::XYT;
using diamond::C;
using diamond::E;
using diamond::H;

// Buffer inputs to kernel.
template <typename T1> struct KernelInputs {
  KernelInputs(T1 dt, T1 *cbuffer, T1 *abslayer, T1 *srclayer, T1 *waveform,
               T1 *zcoeff)
      : dt(dt), cbuffer(cbuffer), abslayer(abslayer), srclayer(srclayer),
        waveform(waveform), zcoeff(zcoeff) {}

  T1 dt;
  T1 *cbuffer, *abslayer, *srclayer, *waveform, *zcoeff;
};

template <typename T> struct KernelInternal {
  KernelInternal(T *buffer, T *cbuffer, T *mask, T *src)
      : buffer(buffer), cbuffer(cbuffer), mask(mask), src(src) {}

  T *buffer, *cbuffer, *mask, *src;
};

// Input parameters for the simulation kernel.
template <typename T, typename T1> struct KernelArgs {
  KernelArgs(RunShape rs, KernelInternal<T> internal, KernelInputs<T1> inputs,
             T1 *output)
      : rs(rs), internal(internal), inputs(inputs), output(output) {}

  RunShape rs;
  KernelInternal<T> internal;
  KernelInputs<T1> inputs;
  T1 *output;
};

// Number of elements for internal storage needed in shared memory.
__dhce__ int SharedElems(RunShape rs) {
  return buffer::SharedElems(rs) + cbuf::SharedElems(rs.block);
}

// Number of elements for internal storage needed in global memory.
template <typename T> __dhce__ int GlobalElems(RunShape rs) {
  return buffer::GlobalElems(rs) +                 //
         cbuf::GlobalElems(rs.domain) +            //
         slice::ZMask<T>::GlobalElems(rs.domain) + //
         (rs.src.type == RunShape::Src::ZSLICE
              ? slice::ZSrc<T>::GlobalElems(rs.domain)
              : slice::YSrc<T>::GlobalElems(rs.domain.x));
}

// Organizes global memory for the kernel.
template <typename T> struct GlobalPtrs {
  __dhce__ GlobalPtrs(T *ptr, RunShape rs)
      : buffer(ptr),                                      //
        cbuffer(buffer + buffer::GlobalElems(rs)),        //
        abslayer(cbuffer + cbuf::GlobalElems(rs.domain)), //
        srclayer(abslayer + slice::ZMask<T>::GlobalElems(rs.domain)) {}

  T *buffer,     // u- and v-buffers for E- and H-field values.
      *cbuffer,  // Material coefficients.
      *abslayer, // Absorption mask.
      *srclayer; // Current source values.
};

// Convert external inputs to the internal format needed by the simulation
// kernel.
template <typename T, typename T1>
__dhce__ void ConvertInputs(KernelArgs<T, T1> args, RunShape rs, int t, UV w,
                            UV b) {
  field::Init<T, T1>(args.output, rs, t, w, b);
  buffer::Init(args.internal.buffer, rs, t, w, b);
  cbuf::Convert(args.inputs.cbuffer, args.internal.cbuffer, rs, rs.pml.zshift,
                t, w, b, args.inputs.abslayer, args.inputs.dt);
  slice::ConvertMask(args.inputs.abslayer, args.internal.mask, rs, t, w, b,
                     args.inputs.dt);
  if (rs.src.type == RunShape::Src::ZSLICE) {
    slice::ConvertZSrc(args.inputs.srclayer, args.internal.src, rs.src.pos, rs,
                       t, w, b);
  } else { // SrcType == RunShape::SourceType::YSLICE.
    slice::ConvertYSrc(args.inputs.srclayer, args.internal.src, rs,
                       rs.pml.zshift, t, w, b);
  }
}

// TODO: Remove.
template <typename T,  // Type used internally in the kernel.
          typename T1, // Type of external input and output buffers.
          int Npml,    // Number of threads dedicated to auxiliary PML values.
          bool WithGlobal = true, //  "With*" template parameters used for
          bool WithShared = true, // benchmarking purposes only.
          bool WithUpdate = true>
__global__ void DoNothingKernel(KernelArgs<T, T1> args) {}

// The FDTD simulation kernel.
template <typename T,  // Type used internally in the kernel.
          typename T1, // Type of external input and output buffers.
          int Npml,    // Number of threads dedicated to auxiliary PML values.
          bool WithGlobal = true, //  "With*" template parameters used for
          bool WithShared = true, // benchmarking purposes only.
          bool WithUpdate = true>
__global__ void SimulationKernel(KernelArgs<T, T1> args) {

  RunShape rs = args.rs;

  // Initialize constants.
  const int t = defs::ThreadPos();
  const UV w = defs::WarpPos();
  const UV b = defs::BlockPos();
  const bool isaux = defs::IsAux(t, rs);
  const int timesteps = defs::NumTimeSteps(rs.out);
  const int numsteps = scanner::NumSteps(timesteps, rs);
  const T neg_dt =
      isaux ? defs::One<T>() : defs::Convert<T, T1>(-args.inputs.dt);

  // Convert inputs to internal formats.
  ConvertInputs(args, rs, t, w, b);

  // Allocate shared memory.
  extern __shared__ uint shmem[];
  T *sbuf = ((T *)shmem);
  T *scbuf = sbuf + buffer::SharedElems(rs);
  for (int i = 0; i < SharedElems(rs); ++i)
    ((T *)shmem)[i] = defs::Zero<T>(); // Clear shared memory.

  // Used for systolic update.
  diamond::Cell<T> cell;
  slice::ZSrc<T> zsrc;
  slice::YSrc<T> ysrc;
  zcoeff::ZCoeff<T> zcoeff;
  slice::ZMask<T> zmask;

  diamond::InitCell<T>(cell, defs::Zero<T>());
  zcoeff::Load(zcoeff, args.inputs.zcoeff, t, rs.pml.n, rs.pml.zshift, isaux);

  // Begin simulation iterations.
  auto grid = cooperative_groups::this_grid();
  grid.sync();

  for (int step = 0; step < numsteps; ++step) {
    UV bufferpos = scanner::BufferPos(step, b, rs);
    XYT domainpos = scanner::DomainPos(step, w, b, rs);

    if (WithUpdate) {
      update::Scale(cell, zmask, zcoeff, isaux, E);
      update::AddSrc(cell, zsrc, ysrc, args.inputs.waveform, domainpos, rs, t);
    }

    if (WithShared) {
      cbuf::LoadShared(cell, scbuf, t, w, rs.block);
      buffer::LoadShared(cell, sbuf, t, w, rs.block, H);
    }

    if (WithGlobal) {
      buffer::LoadGlobal(cell, args.internal.buffer, t, w, b, bufferpos, rs, E);
      cbuf::LoadGlobal(cell, args.internal.cbuffer, t, w,
                       XY(domainpos.x, domainpos.y), rs.domain);
    }

    if (WithUpdate) {
      update::Update(cell, zcoeff, neg_dt, Npml, isaux, E);
    }
    diamond::Shift(cell, H);
    __syncthreads(); // Separates global E-field load and store.

    update::WriteOutput<T, T1>(cell, domainpos, t, args.output, rs.pml.zshift,
                               rs, isaux);

    if (WithShared) {
      buffer::StoreShared(cell, sbuf, t, w, rs.block, E);
    }

    if (WithGlobal) {
      buffer::StoreGlobal(cell, args.internal.buffer, t, w, b, bufferpos, rs,
                          E);
      cbuf::StoreShared(cell, scbuf, t, w, rs.block);
    }

    __syncthreads(); // Separates shared E-field store and load.

    if (WithShared) {
      buffer::LoadShared(cell, sbuf, t, w, rs.block, E);
    }

    if (WithGlobal) {
      buffer::LoadGlobal(cell, args.internal.buffer, t, w, b, bufferpos, rs, H);

      XYT dpnext = scanner::DomainPos(step + 1, w, b, rs);
      zmask.Load(args.internal.mask, XY(dpnext.x, dpnext.y), rs.domain, t);
      if (rs.src.type == RunShape::Src::ZSLICE) {
        zsrc.Load(args.internal.src, XY(dpnext.x, dpnext.y), rs.domain, t);
      } else { // rs.src.type == RunShape::Src::YSLICE.
        ysrc.Load(args.internal.src, dpnext.x, t);
      }
    }

    // H update
    if (WithUpdate) {
      update::Scale(cell, zmask, zcoeff, isaux, H);
      update::Update(cell, zcoeff, neg_dt, Npml, isaux, H);
    }

    diamond::Shift(cell, E);
    diamond::Shift(cell, C);
    __syncthreads(); // Separates global H-field load and store.

    if (WithShared) {
      buffer::StoreShared(cell, sbuf, t, w, rs.block, H);
    }

    if (WithGlobal) {
      buffer::StoreGlobal(cell, args.internal.buffer, t, w, b, bufferpos, rs,
                          H);
    }

    // grid.sync();
    if (step % rs.spacing == 0)
      grid.sync();
    else
      __syncthreads(); // Separates shared H-field store and load.
  }
}

} // namespace kernel

#endif // _KERNEL_H_
