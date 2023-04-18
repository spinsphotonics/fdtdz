# [*fdtd-z*](github.com/spinsphotonics/fdtdz) -- fast, scalable, accessible photonic simulation

At SPINS 💫, our mission is to create the underlying technology to allow photonic designers to harness compute at scale -- enabling a revolution in photonic design.
*fdtd-z*, the first step towards that goal, is an implementation of the [finite-difference time-domain method](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) (FDTD) that is

* **fast**: 100x faster than CPU implementations such as Meep, Lumerical, RSoft, ..., 
* **scalable**: easily runs simulations on hundreds of nodes and beyond,
* **accessible**: open-source, license-free, and universally accessible.

At the core of *fdtd-z* is a heavily-optimized computational kernel that implements the FDTD update
on the GPU in an efficient (systolic) manner.
The kernel is then made available in [JAX](https://github.com/google/jax),
a high-performance machine learning framework that allows for massive distribution across computational nodes,
as well as paving the way for optimization to be easily added later.
All of this is not only open-sourced and permissively licensed,
but you can easily try it for yourself (link coming), even if you don't have your own GPU!
