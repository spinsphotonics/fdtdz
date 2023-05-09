# -*- coding: utf-8 -*-

__all__ = ["__version__", "fdtdz", "residual"]

from .fdtdz_jax import fdtdz
from .residual import residual
from .fdtdz_jax_version import version as __version__
