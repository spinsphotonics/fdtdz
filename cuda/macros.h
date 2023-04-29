#ifndef _MACROS_H_
#define _MACROS_H_

// Shorthand for various kinds of functions.
//
// We adopt a rather primitive coding style where non-trivial classes (i.e.
// classes with any kind of state) are avoided as much as possible. This is done
// to please the `nvcc` compiler which seems to do get confused at times things
// when dealing with such objects (e.g. registers wasted, uses slow, local
// memory accesses, ...).
//
#define __dhce__ __device__ __host__ constexpr
#define __dhsc__ __device__ __host__ static constexpr
#define __dh__ __device__ __host__

#endif // _MACROS_H_
