# [*fdtd-z*](github.com/spinsphotonics/fdtdz) -- fast, scalable, and free photonic simulation
***fdtd-z*** is our first step in *revolutionizing photonic design* by enabling photonic engineers to **harness compute at scale**.

* **fast**: 100x faster than CPU implementations such as Meep, Lumerical, RSoft, ..., 
* **scalable**: easily runs simulations on hundreds of nodes and beyond,
* **free**: open-source, no license-fees!

Give it a test drive [![in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3/welcome-to-fdtd-z.ipynb),
join the chat [![on https://gitter.im/fdtdz/community](https://badges.gitter.im/fdtdz/community.svg)](https://app.gitter.im/#/room/#fdtdz:gitter.im), and read the [whitepaper](paper/paper.pdf).

### API

***fdtd-z*** exposes a low-level API `fdtdz()` to launch FDTD kernels on the GPU while giving the user fine-grained control over values for permittivity, current source, boundary conditions, and more.

In order to optimize the simulation throughput for nanophotonic applications, a number of constraints are placed on the allowable inputs such as limited dimension of the z-axis, simple non-absorbing dielectric materials only, and adiabatic absorption boundaries along the x- and y-axes.
See the `fdtdz()` docstring for all the details.

While it does not include all the bells and whistles, it is *fast*: delivering ~100X speed-up compared to MEEP/Lumerical even on commodity GPUs such as the Nvidia T4; and we're looking forward to building out additional functionality with the open-source photonics community!

That said, don't take our word for it, [try it out for yourself](https://colab.research.google.com/gist/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3/welcome-to-fdtd-z.ipynb)!

### Install

You must have the GPU-version of [JAX](https://github.com/google/jax) installed first, see [instructions here](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).
Once this is done, you can just do `pip install fdtdz` or `!pip install fdtdz` directly from a Colab notebook as is done in the [example notebook](https://colab.research.google.com/gist/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3/welcome-to-fdtd-z.ipynb).

To install from source, please see this [very helpful write-up](https://github.com/spinsphotonics/fdtdz/issues/14#issue-1794133653) by [@johnrollinson](https://github.com/johnrollinson).

## Frequently Asked Questions

### Is the plan to build additional functionality inside the *fdtd-z* repo?

*fdtd-z* is intended to be a super low-level utility in the UNIX philosophy of "just do one thing".
As such, a number of seemingly critical functionality is intentionally missing such as waveguide mode solving, utilities for building dielectric structures, and more.

Many, if not all, of these functionalities already exist in various repositories such as [MEEP](https://github.com/NanoComp/meep) and [SPINS](https://github.com/stanfordnqp/spins-b), but instead of the same code bits being replicated yet again in *fdtd-z* our hope is that the photonics community will build these in a modular, open-source way.

We envision a toolchain of photonics software built upon a common numerical framework that is both scaleable and differentiable.
At the moment [JAX](https://github.com/google/jax) fits the bill.
*fdtd-z* is our contribution to this ecosystem, and we hope there will be many others to join us in building something amazing :).

### Will dispersive materials be supported?

Currently, *fdtd-z* only supports simple dielectric materials (not even absorption is supported) for performance reasons (see the [whitepaper](paper/paper.pdf) for the full story).
In the systolic scheme that *fdtd-z* uses there are two many constraints that prevent us from modeling more complex materials:

* bandwidth limitations: Additional coefficients used in the update equation need to be loaded, while additional auxiliary fields would both need to be loaded and written back to disk. FDTD is already a heavily bandwidth-limited algorithm and this would further decrease performance.
* register pressure: CUDA threads are hard-limited to a maximum of 256 registers (reference needed), which are needed for fast access to E- and H-field values, as well as storing coefficients.
The current implementation of *fdtd-z* already uses the maximum number of registers -- a fundamental change in the basic architecture of the systolic memory system (such as using shared memory instead of registers) would be needed to simulate dispersive materials and other more complex material systems.

For dispersive materials, a simple work-around is to run individual single-frequency simulations at each wavelength of interest with the appropriate modified permittivity.
While more laborious *fdtd-z* is designed to be fast as well as easy to parallelize, this work-around also allows for arbitrarily complex dispersions to be modeled accurately.

### What about more flexible output sources?

Output sources are currently limited to materializing the E-field over the entire simulation domain for a set of equidistant time steps and leaving it to the user to back out frequency components as desired.
While we have not exhaustively tested other output schemes, we can summarize the thinking behind this design decision.

* The bandwidth costs of a continuous, or running, update are extremely high. In addition to having to read and write the E- and H-field values needed for the FDTD update, an update scheme (e.g. performing a rolling DFT) would have to both read and write output values at every frequency of interest as well.
* The systolic update scheme negates some of the advantages of only materializing a sub-volume of the simulation domain -- the whole grid of CUDA threads really needs to move along at the same rate (this is also the reason why PML absorbing conditions are not implemented along the x-y plane since the additional, intially localized, computational cost would get spread across a large portion of the simulation domain). Additionally, the time-skew of the systolic scheme also significantly smears the additional cost initially localized to a single time step so that it is felt across multiple time steps.

For these prinicipal reasons, *fdtd-z* has tried to limit output operations to be write-only and to be as temporally sparse as possible.
That said, we do think there is room for additional flexibility in terms of allowing for (potentially multiple) subdomains to be materialized for a larger number of time steps in order to allow a greater number of frequency components to be inferred from a single simulation.
Please let us know if this would be important for your application!

### Is multi-GPU supported?

While *fdtd-z* is not able to distribute a single simulation across multiple GPUs, building on [JAX](https://github.com/google/jax) means that there should be excellent support readily available for parallelization in terms of distributing multiple simulations across multiple GPUs (where each device has 1 or more simulations to solve).
The `jax.pmap` [documentation](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap) is probably the right starting point for this.

### `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`

*fdtd-z* uses CUDA [cooperative groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups) to implement the systolic scheme outlined in the [whitepaper](paper/paper.pdf) and get around the GPU bandwidth bottleneck.
Because of this, the [launch parameters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy) of the kernel become tightly connected to the underlying architecture of the GPU. 
In particular the `(gridu, gridv)` part of the launch parameters must not exceed the number of streaming multiprocessors (SMs) that are on the GPU.
For example, the RTX4000 has 36 GPUs so it would make sense to use `(gridu, gridv) = (6, 6)` (note that there is the additional constraint that `blocku * gridu <= blockv * gridv`).
If `gridu * gridv` is greater than the number of available GPUs, then an attempt to launch the kernel will result in the `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` error.
