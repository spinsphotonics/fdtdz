import fdtdz_jax
from jax.test_util import check_grads
from jax.config import config
import jax.numpy as jnp
import jax
import pytest
import numpy as np

np.set_printoptions(threshold=np.inf)


# TODO: Tests are missing!!!
def test_fdtdz():
  xx, yy, zz, tt = 16, 16, 44, 100
  epsilon = np.ones((3, xx, yy, zz), np.float32)
  dt = float(0.5)
  source_field = np.zeros((2, 2, xx, yy), np.float32)
  source_field[0, 0, xx // 2, yy // 2] = 1.0
  source_waveform = np.zeros((tt, 2), np.float32)
  source_waveform[0, 0] = 1.0
  source_position = zz // 2
  absorption_mask = np.zeros((3, xx, yy), np.float32)
  pml_kappa = np.ones((zz, 2), np.float32)
  pml_sigma = np.zeros((zz, 2), np.float32)
  pml_widths = (10, 10)
  output_steps = (tt - 10 - 1, tt, 10)
  use_reduced_precision = False
  launch_params = "Quadro RTX 4000"

  out = fdtdz_jax.fdtdz(
      epsilon,
      dt,
      source_field,
      source_waveform,
      source_position,
      absorption_mask,
      pml_kappa,
      pml_sigma,
      pml_widths,
      output_steps,
      use_reduced_precision,
      # launch_params,
      ((2, 2), (2, 2), 2, (7, 5)))

  print(out[0, 0, :, :, zz // 2])
  print(f"Does output contain NaN values? {np.any(np.isnan(out))}")
  print(out.shape)
