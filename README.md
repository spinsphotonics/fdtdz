# [*fdtd-z*](github.com/spinsphotonics/fdtdz) -- fast, scalable, and free photonic simulation
***fdtd-z*** is our first step in *revolutionizing photonic design* by enabling photonic engineers to **harness compute at scale**.

* **fast**: 100x faster than CPU implementations such as Meep, Lumerical, RSoft, ..., 
* **scalable**: easily runs simulations on hundreds of nodes and beyond,
* **free**: open-source, no license-fees!

[Try it now!](https://gist.github.com/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3/welcome-to-fdtd-z.ipynb) and read the [whitepaper](paper/paper.pdf) to understand how it works.

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

### What about multi-frequency output sources?
