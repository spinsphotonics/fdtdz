__all__ = ["residual"]

import functools

import numpy as np


def _diff(field, axis, is_fwd):
  """Forward or backward spatial difference of 3D `field`."""
  if is_fwd:
    return np.roll(field, shift=-1, axis=axis) - field
  else:
    return field - np.roll(field, shift=1, axis=axis)


def _curl(fields, sz, is_fwd):
  """Computes ∇×E, with vector-field `fields` and `sz` cell-size along z."""
  d = functools.partial(_diff, is_fwd=is_fwd)
  return (d(fields[2], axis=1) - d(fields[1], axis=2) / sz,
          d(fields[0], axis=2) / sz - d(fields[2], axis=0),
          d(fields[1], axis=0) - d(fields[0], axis=1))


def _curl_curl(fields, sz):
  """Computes ∇×∇×E, with vector-field `fields` and `sz` cell-size along z."""
  return np.stack(_curl(_curl(fields, sz[:, 1], is_fwd=True), sz[:, 0],
                        is_fwd=False))


def _wave_operator(omega, fields, epsilon, sz, source):
  """Computes (∇×∇× - ω²ε)E - iωJ with `sz` cell-size along z."""
  return (_curl_curl(fields, sz) - epsilon * omega**2 * fields -
          1j * omega * source)


def _frequency_component(out, steps, omega, dt):
  """Returns E-field at `omega` for simulation output `out` at `steps`."""
  theta = omega * dt * steps
  phases = np.stack([np.cos(theta), -np.sin(theta)], axis=-1)
  parts = np.einsum('ij,jk...->ik...', np.linalg.pinv(phases), out)
  return parts[0] + 1j * parts[1]


def _source_amplitude(source_waveform, omega, dt):
  """Returns complex scalar denoting source amplitude at `omega`."""
  theta = omega * dt * (np.arange(source_waveform.shape[0]) - 0.5)
  parts = np.mean(2 * np.stack([np.cos(theta), -np.sin(theta)])[..., None] * source_waveform,
                  axis=1)
  return parts[0] + 1j * parts[1]


def residual(
    omega,
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
):
  """Computes the residual of the wave equation, namely (∇×∇× - ω²ε)E - iωJ. 

  Args:

    omega: Real-valued scalar denoting the angular frequency at which to
      evaluate the residual.

    fields: `(n, 3, xx, yy, zz)`-shaped array of `n` snapshots of the simulated
      E-field at timesteps given by `output_steps`.

    All other arguments are identical to those used in `fdtdz()`.

  Returns:
    `(3, xx, yy, zz)` array of complex floats containing the residual values.

  """
  complex_fields = _frequency_component(
      fields, np.arange(*output_steps), omega, dt)

  eps = epsilon * (1 - 1j * abs_mask[..., None] / omega)

  sz = pml_kappa + pml_sigma / (pml_alpha + 1j * omega)

  amp = (1 / dt) * _source_amplitude(source_waveform, omega, dt)
  src = np.zeros_like(complex_fields)
  if source_field.shape[-1] == 1:  # z-plane source.
    src[(0, 1), :, :, source_position] = (
        amp[0] * source_field[0, (0, 1), :, :, 0] +
        amp[1] * source_field[1, (0, 1), :, :, 0])

  elif source_field.shape[-2] == 1:  # y-plane source.
    src[(0, 2), :, source_position, :] = amp[0] * source_field[:, :, 0, :]
    src[(0, 2), :, source_position-1, :] = amp[1] * source_field[:, :, 0, :]

  else:  # x-plane source.
    src[(1, 2), source_position, :, :] = amp[0] * source_field[:, 0, :, :]
    src[(1, 2), source_position-1, :, :] = amp[1] * source_field[:, 0, :, :]

  return _wave_operator(omega, complex_fields, eps, sz, src)
