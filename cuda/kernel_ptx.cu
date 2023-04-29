// Used to output ptx code.

#include "kernel.h"

template __global__ void
kernel::SimulationKernel<INTERNALTYPE, EXTERNALTYPE, NPML, //
                         WITHGLOBAL, WITHSHARED, WITHUPDATE>(
    KernelArgs<INTERNALTYPE, EXTERNALTYPE> args);

// _ZN6kernel16SimulationKernelI7__half2fLi5ELb1ELb1ELb1EEEvNS_10KernelArgsIT_T0_EE
// _ZN6kernel16SimulationKernelI7__half2fLi3ELb0ELb0ELb1EEEvNS_10KernelArgsIT_T0_EE
// _ZN6kernel16SimulationKernelI7__half2fLi3ELb1ELb1ELb1EEEvNS_10KernelArgsIT_T0_EE
// _ZN6kernel16SimulationKernelIffLi5ELb1ELb1ELb1EEEvNS_10KernelArgsIT_T0_EE
// _ZN6kernel16SimulationKernelIffLi6ELb1ELb1ELb1EEEvNS_10KernelArgsIT_T0_EE
