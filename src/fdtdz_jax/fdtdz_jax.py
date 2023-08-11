__all__ = ["fdtdz"]

import functools
import math
import os
import sys

import numpy as np
import jax
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# If the GPU version exists, also register those
from . import gpu_ops
for _name, _value in gpu_ops.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="gpu")

# Number of cells to use for padding the systolic update scheme.
_NUM_PAD_CELLS = 4


def _preset_launch_params(device_kind):
  """Returns ``(block, grid, spacing, cc)`` parameters for ``device_kind``."""
  if device_kind == "Quadro RTX 4000":
    return ((2, 4), (6, 6), 2, (7, 5))
  elif device_kind == "Tesla T4":
    return ((2, 4), (8, 5), 2, (7, 5))
  elif "V100" in device_kind:
    return ((2, 4), (10, 8), 2, (7, 0))
  elif "A100" in device_kind:
    return ((2, 4), (12, 9), 4, (8, 0))
  else:
    raise ValueError(
        f"No preset launch parameters available for \"{device_kind}\".")


def _padded_domain_shape(shape, launch_params):
  """Returns ``(xx, yy) >= shape`` compatible with ``launch_params``."""
  xx, yy = shape
  (bu, bv), (gu, gv), spacing, _ = launch_params

  # First, we need to make the domain large enough in the x-direction.
  #
  # This requires that the size of the overall diamond (spacing included) along
  # the v-axis not be smaller than ``xx`` with initial padding of 4 on both
  # boundaries included.
  #
  xx = max(xx + 8, (2 * bv + spacing) * gv)

  # Second, we need to make the domain large enough in the y-direction.
  #
  # This requires that, at minimum, the total height of the diamond be less
  # than or equal to ``yy`` with padding of 4 included on both boundaries.
  #
  yy = max(yy + 8, (2 * (bu * gu + bv * gv)))

  # Third, we need to make ``yy`` traversable in an integer amount of scans.
  #
  # This is satisfied as long as ``yy`` is equal to one u-length and a positive
  # integer number of v-lengths of the diamond.
  #
  n = int(math.ceil((yy - (2 * bu * gu)) / (2 * bv * gv)))
  yy = (2 * bu * gu) + n * (2 * bv * gv)

  return (xx, yy)


def _is_source_type(f, type):
  """``True`` iff source field ``f`` is of ``type``."""
  return ((type == "x" and f.ndim == 4 and f.shape[1] == 1) or
          (type == "y" and f.ndim == 4 and f.shape[2] == 1) or
          (type == "z" and f.ndim == 5 and f.shape[4] == 1))


def _flip_roll_component(array, component, splitaxis, flipaxis, scalez=1.0):
  """Flip around ``slipaxis`` and shift ``component`` of ``splitaxis``."""
  array = jnp.flip(array, axis=flipaxis)
  splits = jnp.split(array, array.shape[splitaxis], axis=splitaxis)
  splits[component] = scalez * jnp.roll(splits[component], -1, axis=flipaxis)
  return jnp.concatenate(splits, axis=splitaxis)


def _flip_roll(array, axis):
  """Flip on all axes, shift on ``axis``."""
  return jnp.roll(jnp.flip(array, axis=range(array.ndim)), -1, axis=axis)


@functools.partial(jax.jit, static_argnames=[
    "dt",
    "source_position",
    "pml_widths",
    "output_steps",
    "use_reduced_precision",
    "launch_params",
    "offset",
])
def fdtdz(
    epsilon,
    dt,
    source_field,
    source_waveform,
    source_position,
    absorption_mask,
    pml_kappa,
    pml_sigma,
    pml_alpha,
    pml_widths,
    output_steps,
    use_reduced_precision,
    launch_params=None,
    offset=(0, 0, 0),
):
  """Execute a FDTD simulation.

  ``fdtd-z`` is an implementation of the finite-difference time-domain (FDTD)
  method that is efficiently mapped to the GPU via a systolic update scheme
  [#whitepaper_ref]_. This function exposes, as a JAX primitive, a relatively low-level API
  to the underlying CUDA kernel.

  ``fdtd-z`` targets nanophotonic applications with a heavy prioritization on
  simulation throughput, as insufficient throughput is currently the
  bottleneck in many workflows (such as nanophotonic inverse design). As such,
  the flexibility and features of the engine have been kept to a bare
  minimum in the name of performance:

  - Reduced-precision mode is available and recommended which utilizes
    16-bit floating point internally (input and output arrays are always
    at single-precision) and allows for ~2 times larger z-extent. Requires
    (Nvidia) GPUs of compute capability >= 6.0.

  - Z-extent (``zz``) of the simulation domain is fixed to a specific value to
    allow for a natural mapping to the 32 threads/warp architecture of
    Nvidia GPUs. The required value of ``zz`` is given by either ``128 - p`` or
    ``64 - p`` for the cases of reduced precision or full precision
    respectively, where ``p`` is the total number of PML layers for the
    simulation.

  - PML boundary conditions [#cpml_ref]_ are limited to the z-direction, adiabatic
    aborbing boundaries [#absorber_ref]_ must be used in the x- and y-directions.

  - Absorption (apart from absorbing boundary conditions) and dispersion are
    not implemented. The simulated structure must consist of a real-valued
    permittivity that depends only on the E-field component and spatial
    location.

  - The permeability is fixed at a value of ``1`` everywhere in the simulation
    domain.

  - Only current sources as hyperplanes along the y- and z-axes are
    implemented.

  ``fdtd-z`` uses a "dimensionless" system [#meep_ref]_ where the permittivity and
  permeability of vacuum (and therefore the speed of light), as well as the size
  of the Yee cell are all set to a (dimensionless) value of ``1`` (although the
  size of the cell along the z-axis can be varied).

  .. [#whitepaper_ref]  Lu, Jesse, and Jelena Vuckovic. "fdtd-z: A systolic scheme for GPU-accelerated nanophotonic simulation." https://github.com/spinsphotonics/fdtdz/blob/main/paper/paper.pdf
  .. [#cpml_ref]  Roden, J. Alan, and Stephen D. Gedney. "Convolution PML
     (CPML): An efficient FDTD implementation of the CFS–PML for arbitrary
     media." Microwave and optical technology letters 27.5 (2000): 334-339.
  .. [#absorber_ref]  Oskooi, Ardavan, and Steven G. Johnson. "Distinguishing correct from incorrect PML proposals and a corrected unsplit PML for anisotropic, dispersive media." Journal of Computational Physics 230.7 (2011): 2369-2377.
  .. [#meep_ref]  Oskooi, Ardavan F., et al. "MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method." Computer Physics Communications 181.3 (2010): 687-702.

  Args:
    epsilon: ``(3, xx0, yy0, zz0)``-shaped array of floats placed at offset
      ``offset`` in the simulation domain, representing the permittivity values
      at the ``Ex``, ``Ey``, and ``Ez`` nodes of the Yee cell respectively. We
      use the convention that these components are located at ``(0.5, 0, 0)``,
      ``(0, 0.5, 0)``, and ``(0, 0, 0.5)`` respectively, for a Yee cell of
      side-length 1. To explicitly define permittivity values at every point in
      the simulation, use ``epsilon.shape=(3, xx, yy, zz)`` and
      ``offset=(0, 0, 0)``, otherwise the simulation domain size is inferred
      from the other parameters, with permittivity values outside of ``epsilon``
      inferred via edge padding ``epsilon``.
    dt: Scalar float representing the amount of time elapsed in one update step.
    source_field: An array of shape ``(2, 1, yy, zz)``, ``(2, xx, 1, zz)``, or
      ``(2, 2, xx, yy, 1)``-shaped array for a source at
      ``x = source_position``, ``y = source_position``, or ``z = source_position``
      respectively. The ``(2, 1, yy, zz)`` source features ``Ey`` and ``Ez``
      components in that order, while the ``(2, xx, 1, zz)`` source features ``Ex``
      and ``Ez`` components in that order. The ``(2, 2, xx, yy, 1)`` allows for the
      specification of two separate source fields at ``[0, :, :, :, :]`` and
      ``[1, :, :, :, :]`` which each contain ``Ex`` and ``Ey`` components in that
      order.
    source_waveform: ``(tt, 2)``-shaped array of floats denoting the temporal
      variation to apply to the each of the source fields, where ``tt`` is the
      total number of update steps needed (note that the first update occurs at
      step ``0``). Specifically, the subarray at ``(tt, i)`` applies a temporal
      variation to the ``(i, 2, xx, yy. 1)`` subarray of a source at
      ``z = source_position``, while for a source at ``x = source_position`` or
      ``y = source_position`` the temporal variation is applied to the source
      field at ``source_position - i``.
    source_position: integer representing the position of the source along
      either the y- or z-axes. For the case of a source at ``y = source_position``
      the source field is applied (with the corresponding waveform) at both
      ``y = source_position`` and ``y = source_position - 1``, with the additional
      performance constraint that ``source_position`` must be even.
    absorption_mask: ``(3, xx, yy)``-shaped array of floats representing a
      z-invariant conductivity intended to allow for adiabatic absorbing
      boundary conditions along the x- and y-axes according to
      ``conductivity(x, y, z) = aborption_mask(x, y) * epsilon(x, y, z)``
      for the ``Ex``, ``Ey``, and ``Ez`` component respectively.
    pml_kappa: ``(zz, 2)``-shaped array of floats denoting the distance along the
      z-axis between adjacent Yee cells. This is primarily intended to be used
      as a stretching parameter for the PML, but equivalently also determines
      the unit cell length along the z-axis throughout the simulation domain.
      Specifically, ``(zz, 0)`` represents the distance between successive layers
      of ``Ex``, ``Ey``, and ``Hz`` nodes, while ``(zz, 1)`` represents the distance
      between layers of ``Hx``, ``Hy``, and ``Ez`` nodes.
    pml_sigma: ``(zz, 2)``-shaped array of floats for the conductivity of the PML
      region, where ``(zz, 0)`` and ``(zz, 1)`` are the values at the
      (``Ex``, ``Ey``, ``Hz``) and (``Hx``, ``Hy``, ``Ez``) layers respectively. Must be set
      to ``0`` outside of the PML regions.
    pml_alpha: ``(zz, 2)``-shaped array of floats similar to ``pml_sigma``. Must
      also be set to ``0`` outside of the PML regions.
    pml_widths: ``(bot, top)`` integers specifying the number of cells which
      are to be designated as PML layers at the bottom and top of the
      simulation respectively. For performance reasons, the total number of PML
      layers used in the simulation (``bot + top``) is required to be a multiple
      of ``4``.
    output_steps: ``(start, stop, interval)`` tuple of integers denoting the
      update step at which to start recording output fields, the number of
      update steps separating successive output fields, and the step at which
      to stop recording (not included).
    use_reduced_precision: If ``True``, uses 16-bit (IEEE 754) precision for the
      simulation which allows for a maximum of 128 cells along the z-axis.
      Otherwise, uses 32-bit single-precision with a maximum of 64 cells
      along the z-axis. Both inputs and results are always expected as 32-bit
      arrays.
    launch_params: Integers as an object in the form of
      ``((blocku, blockv), (gridu, gridv), spacing, (cc_major, cc_minor))``,
      specifying the structure of the systolic update to use on the GPU where
      ``(blocku, blockv)`` determines the layout of warps in the u- and
      v-directions within a block and should be ``(2, 4)`` or ``(4, 2)``;
      ``(gridu, gridv)`` specify the layout of blocks on the GPU and must be
      equal to or less than the number of streaming multiprocessors on the
      GPU because of the need for grid-wide synchronization;
      ``spacing`` controls the number of buffers used between each block and
      its downstream neighbor and should be tuned to balance between
      reducing grid synchronization overhead and staying within the limits
      of the L2 cache; and
      ``(cc_major, cc_minor)`` major and minor compute capability of the
      device. Used to determine which precompiled kernel to use. Currently
      allowed values are ``(3, 7)``, ``(6, 0)``, ``(7, 0)``, ``(7, 5)``, and
      ``(8, 0)``. Recommended to use the latest compute capability kernel
      possible that does not exceed the compute capability of the device.
      Use ``None`` to try to use a default value based on
      ``jax.devices()[0].device_kind``.
    offset: ``(x0, y0, z0)`` integers denoting the placement of ``epsilon``
      as well as the desired output subvolume in the simulation domain.

  Returns:
    ``(n, 3, xx0, yy0, zz0)`` array of field values in the subdomain defined
    by ``epsilon.shape`` and ``offset`` representing ``n`` output fields,
    where each output field consists of the values of the ``Ex``, ``Ey``, and
    ``Ez`` node (in that order) at the update steps given by ``output_steps``.

  """
  if launch_params is None:
    launch_params = _preset_launch_params(jax.devices()[0].device_kind)

  total_pml_width = pml_widths[0] + pml_widths[1]

  if not (total_pml_width % 4 == 0 and
          ((use_reduced_precision and total_pml_width <= 40) or
           ((not use_reduced_precision) and total_pml_width <= 20))):
    raise ValueError(
        f"Invalid value for pml_widths, {pml_widths}. "
        "The sum of pml_widths must be a multiple of 4, as well as have a "
        "maximum value of 40 (when using reduced precision) or 20 (otherwise)."
    )

  zz = ((128 if use_reduced_precision else 64) -
        (pml_widths[0] + pml_widths[1]))
  _, xx, yy = absorption_mask.shape

  if not (epsilon.ndim == 4 and epsilon.shape[0] == 3 and
          epsilon.shape[1] + offset[0] <= xx and
          epsilon.shape[2] + offset[1] <= yy and
          epsilon.shape[3] + offset[2] <= zz):
    raise ValueError(f"``epsilon`` must have shape (3, xx0, yy0, zz0) which "
                     f"fits inside a domain of shape {(xx, yy, zz)}, "
                     f"but instead got ``epsilon.shape`` of {epsilon.shape} "
                     f"with ``offset`` of {offset}.")

  if not ((_is_source_type(source_field, "x") and
           source_field.shape == (2, 1, yy, zz)) or
          (_is_source_type(source_field, "y") and
           source_field.shape == (2, xx, 1, zz)) or
          (_is_source_type(source_field, "z") and
           source_field.shape == (2, 2, xx, yy, 1))):
    raise ValueError(f"Invalid source_field shape, got {source_field.shape}.")

  if not ((_is_source_type(source_field, "x") and 0 <= source_position < xx) or
          (_is_source_type(source_field, "y") and 0 <= source_position < yy) or
          (_is_source_type(source_field, "z") and 0 <= source_position < zz)):
    raise ValueError(
        f"Invalid source_position, must be within simulation domain but got "
        f"a value of {source_position}.")

  # Rotate about the ``(x, y) = (1, 1)`` axis to transform into a y-plane source.
  if _is_source_type(source_field, "x"):
    epsilon = jnp.transpose(epsilon, axes=(0, 2, 1, 3))
    epsilon = epsilon[(1, 0, 2), ...]
    epsilon = _flip_roll_component(
        epsilon, component=2, splitaxis=0, flipaxis=3)

    source_field = jnp.swapaxes(source_field, 1, 2)
    source_field = _flip_roll_component(source_field, component=1,
                                        splitaxis=0, flipaxis=3)

    absorption_mask = jnp.transpose(absorption_mask, axes=(0, 2, 1))
    absorption_mask = absorption_mask[(1, 0, 2), ...]

    pml_kappa = _flip_roll(pml_kappa, axis=0)
    pml_sigma = _flip_roll(pml_sigma, axis=0)
    pml_alpha = _flip_roll(pml_alpha, axis=0)

    pml_widths = (pml_widths[0] - 1, pml_widths[1] + 1)

    if epsilon.shape[3] == zz:
      offset = (offset[1], offset[0], offset[2])
    else:
      if offset[2] == 0:
        raise ValueError(f"``offset[2] == 0`` is not supported for "
                         f"``epsilon.shape[3] < zz`` and "
                         f"``source_field.shape == (2, 1, yy, zz)``.")
      offset = (offset[1], offset[0], zz - (offset[2] + epsilon.shape[3]) - 1)

    if epsilon.shape[3] < zz:
      # Unfortunately, because we selectively roll the Ez values only after the
      # simulation, we need an extra layer of values along z.
      epsilon = jnp.pad(epsilon, ((0, 0), (0, 0), (0, 0), (1, 0)), mode="edge")

    try:
      out = fdtdz(
          epsilon,
          dt,
          source_field,
          source_waveform,
          source_position,
          absorption_mask,
          pml_kappa,
          pml_sigma,
          pml_alpha,
          pml_widths,
          output_steps,
          use_reduced_precision,
          launch_params,
          offset,
      )
    except ValueError as e:
      # Make a note that this occurred while doing the transform
      raise ValueError(
          "Note that the exception occurred while running the transformed "
          "problem for a source at ``x = source_position``.") from e

    # Un-rotate outputs.
    out = jnp.transpose(out, axes=(0, 1, 3, 2, 4))
    out = out[:, (1, 0, 2), ...]
    out = _flip_roll_component(
        out, component=2, splitaxis=1, flipaxis=4, scalez=-1)
    if epsilon.shape[3] < zz:
      # Need to eliminate the extra subvolume along z.
      out = out[..., :-1]
    return out

  # TODO: Consider putting this shape logic and shape checking above the "x"
  # transposition code.
  out_start, out_stop, out_interval = output_steps
  out_num = len(range(out_start, out_stop, out_interval))
  tt = out_start + out_interval * (out_num - 1) + 1

  if not (source_waveform.ndim == 2 and source_waveform.shape == (tt, 2)):
    raise ValueError(
        f"source_waveform must be of shape (tt, 2) = ({tt}, 2), but got "
        f"{source_waveform.shape} instead.")

  if not (pml_kappa.ndim == 2 and pml_kappa.shape == (zz, 2)
          and pml_kappa.shape == pml_sigma.shape
          and pml_kappa.shape == pml_alpha.shape):
    raise ValueError(
        f"pml_kappa, pml_sigma, and pml_alpha must all have shape "
        f"(zz, 2) =  ({zz}, 2), but got shapes {pml_kappa.shape}, "
        f"{pml_sigma.shape}, and {pml_alpha.shape} respectively instead.")

  if isinstance(launch_params, str):
    launch_params = _preset_launch_params(launch_params)
  block, grid, spacing, compute_capability = launch_params

  if not (block[0] * grid[0] <= block[1] * grid[1]):
    raise ValueError(
        f"launch_params must have ``(blocku, blockv)`` and ``(gridu, gridv)`` "
        "be \"v-dominant\" in that ``blocku * gridu <= blockv * gridv`` must be "
        "satisfied.")

  if compute_capability not in ((3, 7), (6, 0), (7, 0), (7, 5), (8, 0)):
    raise ValueError(
        f"Unrecognized compute capability {compute_capability}. "
        "Must be one of (3, 7), (6, 0), (7, 0), (7, 5), or (8, 0).")

  # Once again, for performance reasons, we require that there be 4 cells of
  # padding in the x- and y-directions, and that the actual simulated areas
  # domain shape abide by rules related to ``launch_params``.
  pxx, pyy = _padded_domain_shape((xx, yy), launch_params)

  # TODO: Remove when this is in cuda code.
  # TODO: Now need to move this to cuda code.
  # denom = 1 / dt + absorption_mask / 2
  # cbuffer = 1 / (epsilon * denom[:,
  #                                offset[0]:offset[0] + epsilon.shape[1],
  #                                offset[1]:offset[1] + epsilon.shape[2],
  #                                None])
  # abslayer = ((1 / dt) - (absorption_mask / 2)) / denom

  # Convert to update coefficient, will be scaled by the absorption mask inside
  # the simulation kernel.
  cbuffer = 1 / epsilon

  # PML coefficients.
  pml_b = jnp.exp(-((pml_sigma / pml_kappa) + pml_alpha) * dt)
  pml_z = 1 / pml_kappa
  # Avoid division-by-zero.
  pml_a_denom = pml_sigma * pml_kappa + pml_alpha * pml_kappa**2
  pml_a = ((pml_b - 1) * jnp.where(pml_a_denom == 0, 1, pml_sigma) /
           jnp.where(pml_a_denom == 0, pml_kappa, pml_a_denom))

  npml = total_pml_width // (4 if use_reduced_precision else 2)

  # Source position must be modified to account for either auxiliary PML cells
  # or additional padding.
  if _is_source_type(source_field, "z"):
    srcpos = source_position + pml_widths[1]
  else:
    srcpos = source_position + _NUM_PAD_CELLS

  dirname = os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                         "ptx")

  kwargs = {
      "dt": dt,
      "capability": compute_capability[0] * 10 + compute_capability[1],
      "blocku": block[0],
      "blockv": block[1],
      "gridu": grid[0],
      "gridv": grid[1],
      "blockspacing": spacing,
      "domainx": pxx,
      "domainy": pyy,
      "domainz": zz,
      "npml": npml,
      "zshift": pml_widths[1],
      "srctype": 1 if _is_source_type(source_field, "z") else 0,
      "srcpos": srcpos,
      "outstart": out_start,
      "outinterval": out_interval,
      "outnum": out_num,
      "dirname": dirname,
      "withglobal": True,
      "withshared": True,
      "withupdate": True,
      "use_reduced_precision": use_reduced_precision,
      "subvolume_offset": offset,
      "subvolume_size": epsilon.shape[1:],
      "volume_size": absorption_mask.shape[1:3] + (zz,),
  }

  abslayer = jnp.pad(absorption_mask,
                     ((0, 0),
                      (_NUM_PAD_CELLS, pxx - xx - _NUM_PAD_CELLS),
                      (_NUM_PAD_CELLS, pyy - yy - _NUM_PAD_CELLS)))

  # Note that there should be no x-source at this point.
  if _is_source_type(source_field, "y"):
    srclayer = jnp.pad(source_field[:, :, 0, :],
                       ((0, 0),
                        (_NUM_PAD_CELLS, pxx - xx - _NUM_PAD_CELLS),
                        (0, 0)))
  else:  # _is_source_type(source_field, "z").
    srclayer = jnp.pad(source_field[:, :, :, :, 0],
                       ((0, 0),
                        (0, 0),
                        (_NUM_PAD_CELLS, pxx - xx - _NUM_PAD_CELLS),
                        (_NUM_PAD_CELLS, pyy - yy - _NUM_PAD_CELLS)))

  zcoeff = jnp.stack(
      [
          pml_a[:, 0],
          pml_b[:, 0],
          pml_z[:, 0],
          pml_a[:, 1],
          pml_b[:, 1],
          pml_z[:, 1],
      ],
      axis=-1)

  out = fdtdz_impl(cbuffer, abslayer, srclayer, source_waveform, zcoeff,
                   **kwargs)
  return out[:,
             :,
             _NUM_PAD_CELLS:-_NUM_PAD_CELLS,
             _NUM_PAD_CELLS:-_NUM_PAD_CELLS,
             :]


def fdtdz_impl(cbuffer, abslayer, srclayer, waveform, zcoeff, **kwargs):
  """Low-level API to the simulation kernel."""
  buffer_, cbuffer_, mask_, src_, output_ = _fdtdz_prim.bind(
      cbuffer, abslayer, srclayer, waveform, zcoeff, **kwargs)
  # Only return ``output_``, since all other "*_" are temporary arrays needed
  # inside the simulation kernel only.
  return output_


def _internal_shapes(**kwargs):
  """Shapes of temporary (and output) arrays needed during the simulation.

  NOTE: Spatial dimensions correspond to the internal dimensions needed by the
  kernel, _not_ to those provided in the more user-friendly ``fdtdz()``.

  """
  xx, yy, zz = kwargs["domainx"], kwargs["domainy"], kwargs["domainz"]
  return {
      "buffer": (6, xx, yy, 64),  # Larger than needed.
      "cbuffer": (3, xx, yy, 64),  # Larger than needed.
      "mask": (xx, yy // 2, 32),
      "src": (xx, yy // 2, 32) if kwargs["srctype"] == 1 else (2, xx, zz),
      "output": (kwargs["outnum"],
                 3,
                 kwargs["subvolume_size"][0] + 2 * _NUM_PAD_CELLS,
                 kwargs["subvolume_size"][1] + 2 * _NUM_PAD_CELLS,
                 kwargs["subvolume_size"][2]),
  }


def _fdtdz_abstract(cbuffer, abslayer, srclayer, waveform, zcoeff, **kwargs):
  # Assume ``cbuffer`` to be of shape ``(3, xx, yy, zz)``.
  # (_, xx, yy, zz) = cbuffer.shape
  shapes = _internal_shapes(**kwargs)
  return (
      ShapedArray(shapes["buffer"], dtype=jnp.float32),  # buffer.
      ShapedArray(shapes["cbuffer"], dtype=jnp.float32),  # cbuffer.
      ShapedArray(shapes["mask"], dtype=jnp.float32),  # mask.
      ShapedArray(shapes["src"], dtype=jnp.float32),  # src.
      ShapedArray(shapes["output"], dtype=jnp.float32),  # output.
  )


def _default_layouts(*shapes):
  return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _fdtdz_lowering(ctx, cbuffer, abslayer, srclayer, waveform, zcoeff,
                    **kwargs):
  opaque = gpu_ops.build_kernel_descriptor(
      kwargs["dt"],
      kwargs["capability"],
      kwargs["withglobal"],
      kwargs["withshared"],
      kwargs["withupdate"],
      kwargs["blocku"],
      kwargs["blockv"],
      kwargs["gridu"],
      kwargs["gridv"],
      kwargs["blockspacing"],
      kwargs["domainx"],
      kwargs["domainy"],
      kwargs["npml"],
      kwargs["zshift"],
      kwargs["srctype"],
      kwargs["srcpos"],
      kwargs["outstart"],
      kwargs["outinterval"],
      kwargs["subvolume_offset"][0],
      kwargs["subvolume_offset"][1],
      kwargs["subvolume_offset"][2],
      kwargs["subvolume_size"][0],
      kwargs["subvolume_size"][1],
      kwargs["subvolume_size"][2],
      kwargs["volume_size"][0],
      kwargs["volume_size"][1],
      kwargs["volume_size"][2],
      kwargs["outnum"],
      kwargs["dirname"],
  )

  (_, xx, yy, zz) = mlir.ir.RankedTensorType(cbuffer.type).shape
  shapes = _internal_shapes(**kwargs)

  out = custom_call(
      "kernel_f16" if kwargs["use_reduced_precision"] else "kernel_f32",
      out_types=[
          mlir.ir.RankedTensorType.get(shapes["buffer"],
                                       mlir.ir.F32Type.get()),
          mlir.ir.RankedTensorType.get(shapes["cbuffer"],
                                       mlir.ir.F32Type.get()),
          mlir.ir.RankedTensorType.get(shapes["mask"], mlir.ir.F32Type.get()),
          mlir.ir.RankedTensorType.get(shapes["src"], mlir.ir.F32Type.get()),
          mlir.ir.RankedTensorType.get(shapes["output"],
                                       mlir.ir.F32Type.get()),
      ],
      operands=[cbuffer, abslayer, srclayer, waveform, zcoeff],
      operand_layouts=[
          (3, 2, 1, 0),  # cbuffer.
          (1, 2, 0),  # abslayer.
          (2, 0, 1) if kwargs["srctype"] == 0 else (1, 2, 3, 0),  # srclayer.
          (1, 0),  # waveform.
          (1, 0),  # zcoeff.
      ],
      result_layouts=_default_layouts(shapes["buffer"], shapes["cbuffer"],
                                      shapes["mask"], shapes["src"],
                                      shapes["output"]),
      backend_config=opaque)

  return out


# Register op with JAX.
_fdtdz_prim = core.Primitive("fdtdz")
_fdtdz_prim.multiple_results = True
_fdtdz_prim.def_impl(functools.partial(xla.apply_primitive, _fdtdz_prim))
_fdtdz_prim.def_abstract_eval(_fdtdz_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_fdtdz_prim, _fdtdz_lowering, platform="gpu")
