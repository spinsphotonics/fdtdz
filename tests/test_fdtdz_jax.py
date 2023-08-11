# autopep8: off

import os

# Needed for large simulation tests.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import fdtdz_jax
from jax.test_util import check_grads
from jax.config import config
import jax.numpy as jnp
import jax
import pytest
import numpy as np

# autopep8: on


def _ramped_sin(wavelength, ramp, dt, tt, delay=4):
  """Sine function with a gradual ramp."""
  t = (2 * np.pi * dt / wavelength) * np.arange(tt, dtype=np.float32)
  return ((1 + np.tanh(t / ramp - delay)) / 2) * np.sin(t)


def _absorption_profiles(numcells, width, smoothness):
  """1D quadratic profile for adiabatic absorption boundary conditions."""
  center = (numcells - 1) / 2
  offset = np.array([[0], [0.5]], dtype=np.float32)
  pos = np.arange(numcells, dtype=np.float32) + offset
  pos = np.abs(pos - center) - center + width
  pos = np.clip(pos, a_min=0, a_max=None)
  return smoothness * np.power(pos, 2)


def _cross_profiles(x, y):
  """Combine two 1D absorption profiles into a 2D profile."""
  return np.max(np.meshgrid(x, y, indexing='ij'), axis=0, keepdims=True)


def _absorption_mask(xx, yy, width, smoothness):
  """Adiabatic absorption boundary condition in the x-y plane."""
  x = _absorption_profiles(xx, width, smoothness)
  y = _absorption_profiles(yy, width, smoothness)
  return np.concatenate([_cross_profiles(x[0], y[1]),
                         _cross_profiles(x[1], y[0]),
                         _cross_profiles(x[1], y[1])])


def _safe_div(x, y):
  return np.zeros_like(x) if y == 0 else x / y


def _pml_sigma_values(pml_widths, zz, ln_R=16.0, m=4.0):
  """Conductivity values for PML boundary condition along the z-axis."""
  offset = np.array([[0], [0.5]], dtype=np.float32)
  z = np.arange(zz, dtype=np.float32) + offset
  z = np.stack([_safe_div(pml_widths[0] - z, pml_widths[0]),
                _safe_div(z + 0.5 - zz + pml_widths[1], pml_widths[1])], axis=-1)
  z = np.max(np.clip(z, a_min=0, a_max=None), axis=-1)
  return ((m + 1) * ln_R * z**m).T


def _simulate(epsilon, dt, tt, src_type, src_wavelength, src_ramp, abs_width,
              abs_smoothness, pml_widths, output_steps, use_reduced_precision,
              subvolume_offset=(0, 0, 0), subvolume_size=None):
  """Run a simple continuous-wave dipole-source simulation."""
  xx, yy = epsilon.shape[1:3]
  zz = (128 if use_reduced_precision else 64) - sum(pml_widths)
  # TODO: Put some structure in ``epsilon``.
  # epsilon = np.ones((3, xx, yy, zz), np.float32)

  abs_mask = _absorption_mask(xx, yy, abs_width, abs_smoothness)
  pml_kappa = np.ones((zz, 2), np.float32)
  pml_sigma = _pml_sigma_values(pml_widths, zz)
  pml_alpha = 0.05 * np.ones((zz, 2), np.float32)

  if src_type == "z":
    source_field = np.zeros((2, 2, xx, yy, 1), np.float32)
    source_field[:, 0, xx // 2, yy // 2, 0] = [2, -1]
    source_position = (zz - sum(pml_widths)) // 2 + pml_widths[0]

  elif src_type == "y":
    source_field = np.zeros((2, xx, 1, zz), np.float32)
    source_field[0, xx // 2, 0, zz // 2] = 1
    source_position = yy // 2

  elif src_type == "x":
    source_field = np.zeros((2, 1, yy, zz), np.float32)
    source_field[0, 0, yy // 2,  zz // 2] = 1
    source_position = xx // 2

  else:
    raise ValueError("Did not recognize `src_type`.")

  source_waveform = np.broadcast_to(
      _ramped_sin(src_wavelength, src_ramp, dt, tt)[:, None], (tt, 2))

  if subvolume_size is None:
    sim_epsilon = epsilon
  else:
    sim_epsilon = epsilon[
        :,
        subvolume_offset[0]:subvolume_offset[0] + subvolume_size[0],
        subvolume_offset[1]:subvolume_offset[1] + subvolume_size[1],
        subvolume_offset[2]:subvolume_offset[2] + subvolume_size[2]]

  fields = fdtdz_jax.fdtdz(
      sim_epsilon,
      dt,
      source_field,
      source_waveform,
      source_position,
      abs_mask,
      pml_kappa,
      pml_sigma,
      pml_alpha,
      pml_widths,
      output_steps,
      use_reduced_precision,
      launch_params=jax.devices()[0].device_kind,
      offset=subvolume_offset,
  )

  if subvolume_size is None:
    # Measuring error is only relevant for the full output domain.
    err = fdtdz_jax.residual(
        2 * np.pi / src_wavelength,
        fields,
        epsilon,
        dt,
        source_field,
        source_waveform,
        source_position,
        abs_mask,
        pml_kappa,
        pml_sigma,
        pml_alpha,
        pml_widths,
        output_steps,
    )

    return fields, err
  else:
    return fields, None


@pytest.mark.parametrize("src_type", ["x", "y", "z"])
@pytest.mark.parametrize(
    "xx,yy,tt,dt,src_wavelength,use_reduced_precision,max_err",
    [(200, 200, 20000, 0.5, 10.0, True, 2e-2),
     (200, 200, 40000, 0.25, 10.0, True, 2e-2),
     (200, 200, 10000, 0.55, 7.8, True, 2e-2),
     ])
def test_point_source(xx, yy, tt, dt, src_type, src_wavelength,
                      use_reduced_precision, max_err):
  quarter_period = int(round(src_wavelength / 4 / dt))

  def mysim(use_subvolume=False):
    pml_widths = (20, 20)
    zz = (128 if use_reduced_precision else 64) - sum(pml_widths)
    epsilon = np.ones((3, xx, yy, zz), np.float32)
    return _simulate(
        epsilon=epsilon,
        dt=dt,
        tt=tt,
        src_type=src_type,
        src_wavelength=src_wavelength,
        src_ramp=12,
        abs_width=40,
        abs_smoothness=1e-3,
        pml_widths=(20, 20),
        output_steps=(tt - quarter_period - 1, tt, quarter_period),
        use_reduced_precision=use_reduced_precision,
        subvolume_offset=(120, 130, 30) if use_subvolume else (0, 0, 0),
        subvolume_size=(5, 10, 3) if use_subvolume else None,
    )

  full_fields, err = mysim()
  assert not np.any(np.isnan(full_fields))
  assert np.max(np.abs(err)) < max_err

  sub_fields, err = mysim(use_subvolume=True)
  np.testing.assert_array_equal(
      full_fields[..., 120:125, 130:140, 30:33], sub_fields)


@pytest.mark.parametrize(
    "xx0,yy0,zz0,num_output",
    [(200, 200, 10, 1),
     (400, 400, 10, 1),
     (800, 800, 10, 1),
     (1200, 1200, 10, 1),
     (1300, 1300, 10, 1),
     (1300, 1300, 10, 6),
     ])
def test_large_sim(
        xx0, yy0, zz0, num_output, dt=0.5, tt=1000, pml_widths=(8, 8),
        use_reduced_precision=True, abs_width=50, abs_smoothness=1e-2,
        src_wavelength=10.0, src_ramp=4):
  """Run a large simulation using the subvolume feature."""
  epsilon = jnp.ones((3, xx0, yy0, zz0))

  xx, yy = xx0 + 2 * abs_width, yy0 + 2 * abs_width
  zz = (128 if use_reduced_precision else 64) - sum(pml_widths)

  abs_mask = _absorption_mask(xx, yy, abs_width, abs_smoothness)
  pml_kappa = jnp.ones((zz, 2))
  pml_sigma = _pml_sigma_values(pml_widths, zz)
  pml_alpha = 0.05 * jnp.ones((zz, 2))

  # y-source.
  source_field = jnp.zeros((2, xx, 1, zz))
  source_field = source_field.at[0, xx // 2, 0, zz // 2].set(1.0)
  source_position = yy // 2

  source_waveform = jnp.broadcast_to(
      _ramped_sin(src_wavelength, src_ramp, dt, tt)[:, None], (tt, 2))

  output_steps = (tt - num_output, tt, 1)

  fields = fdtdz_jax.fdtdz(
      epsilon,
      dt,
      source_field,
      source_waveform,
      source_position,
      abs_mask,
      pml_kappa,
      pml_sigma,
      pml_alpha,
      pml_widths,
      output_steps,
      use_reduced_precision,
      offset=(abs_width, abs_width, zz // 2 - zz0 // 2),
  )
  assert jnp.sum(jnp.abs(fields)) > 0


def test_raises_correct_exception():
  use_reduced_precision = True
  tt = 10
  pml_widths = (10, 10)
  xx = 100
  yy = 120
  zz = (128 if use_reduced_precision else 64) - sum(pml_widths)

  # `pml_widths` not a multiple of 4.
  with pytest.raises(ValueError, match="The sum of pml_widths"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=(10, 11),
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # Invalid shape for `epsilon`.
  with pytest.raises(ValueError, match="``epsilon`` must have shape"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz+1), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # Invalid `x = source_position` source.
  with pytest.raises(ValueError, match="Invalid source_field shape"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 1, yy+1, zz), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=xx//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # Invalid `y = source_position` source.
  with pytest.raises(ValueError, match="Invalid source_field shape"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, xx, 1, zz-1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=yy//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # Invalid `z = source_position` source.
  with pytest.raises(ValueError, match="Invalid source_field shape"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy+1, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # `source_position` out of bounds.
  with pytest.raises(ValueError, match="Invalid source_position"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # `source_waveform` too short.
  with pytest.raises(ValueError, match="source_waveform must be of shape"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt-1, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # `pml_kappa` wrong shape.
  with pytest.raises(ValueError, match="pml_kappa, pml_sigma, and pml_alpha"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz-1, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # `pml_sigma` wrong shape.
  with pytest.raises(ValueError, match="pml_kappa, pml_sigma, and pml_alpha"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz-1, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # `pml_alpha` wrong shape.
  with pytest.raises(ValueError, match="pml_kappa, pml_sigma, and pml_alpha"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz-1, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # `launch_params` not v-dominant.
  with pytest.raises(ValueError, match="launch_params must have"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((4, 2), (2, 2), 2, (7, 5)),
        use_reduced_precision=True)

  # Invalid compute capability.
  with pytest.raises(ValueError, match="Unrecognized compute capability"):
    fdtdz_jax.fdtdz(
        epsilon=np.ones((3, xx, yy, zz), np.float32),
        dt=0.5,
        source_field=np.zeros((2, 2, xx, yy, 1), np.float32),
        source_waveform=np.zeros((tt, 2), np.float32),
        source_position=zz//2,
        absorption_mask=np.ones((3, xx, yy), np.float32),
        pml_kappa=np.ones((zz, 2), np.float32),
        pml_sigma=np.zeros((zz, 2), np.float32),
        pml_alpha=np.zeros((zz, 2), np.float32),
        pml_widths=pml_widths,
        output_steps=(tt - 1, tt, 1),
        launch_params=((2, 2), (2, 2), 2, (0, 5)),
        use_reduced_precision=True)
