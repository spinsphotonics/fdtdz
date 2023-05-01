# [*fdtd-z*](github.com/spinsphotonics/fdtdz) -- fast, scalable, and free photonic simulation
***fdtd-z*** is our first step in *revolutionizing photonic design* by enabling photonic engineers to **harness compute at scale**.

* **fast**: 100x faster than CPU implementations such as Meep, Lumerical, RSoft, ..., 
* **scalable**: easily runs simulations on hundreds of nodes and beyond,
* **free**: open-source, no license-fees!

[Try it now!](https://gist.github.com/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/jlu-spins/0f3c5459bd4386150ae30b17f7c6a5e3/welcome-to-fdtd-z.ipynb)

## Frequently Asked Questions

### Is the plan to build additional functionality inside the *fdtd-z* repo?

*fdtd-z* is intended to be a super low-level utility in the UNIX philosophy of "just do one thing".
As such, a number of seemingly critical functionality is intentionally missing such as waveguide mode solving, utilities for building dielectric structures, and more.

Many, if not all, of these functionalities already exist in various repositories such as [MEEP](https://github.com/NanoComp/meep) and [SPINS](https://github.com/stanfordnqp/spins-b), but instead of the same code bits being replicated yet again in *fdtd-z* our hope is that the photonics community will build these in a modular, open-source way.

We envision a toolchain of photonics software built upon a common numerical framework that is both scaleable and differentiable.
At the moment [JAX](https://github.com/google/jax) fits the bill.
*fdtd-z* is our contribution to this ecosystem, and we hope there will be many others to join us in building something amazing :).

### I am just wondering when the use of dispersive materials such as Silicon will be possible.

### What about multi-frequency output sources?
