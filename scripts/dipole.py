import fdtdz_jax
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
import time

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")


def ramped_sin(omega, width, delay, dt, tt):
    t = omega * np.arange(tt, dtype=np.float32)
    return ((1 + np.tanh(t / width - delay)) / 2) * np.sin(t)


def absorption_profiles(numcells, width, smoothness):
    center = (numcells - 1) / 2
    offset = np.array([[0], [0.5]], dtype=np.float32)
    pos = np.arange(numcells, dtype=np.float32) + offset
    pos = np.abs(pos - center) - center + width
    pos = np.clip(pos, a_min=0, a_max=None)
    return smoothness * np.power(pos, 2)


def cross_profiles(x, y):
    return np.max(np.meshgrid(x, y, indexing="ij"), axis=0, keepdims=True)


def absorption_mask(xx, yy, width, smoothness):
    x = absorption_profiles(xx, width, smoothness)
    y = absorption_profiles(yy, width, smoothness)
    return np.concatenate([cross_profiles(x[0], y[1]),
                           cross_profiles(x[1], y[0]),
                           cross_profiles(x[1], y[1])])


def safe_div(x, y):
    return np.zeros_like(x) if y == 0 else x / y


def pml_sigma_values(pml_widths, zz, ln_R=0.5, m=1.3):
    offset = np.array([[0], [0.5]], dtype=np.float32)
    z = np.arange(zz, dtype=np.float32) + offset
    z = np.stack([safe_div(pml_widths[0] - z, pml_widths[0]),
                  safe_div(z + 0.5 - zz + pml_widths[1], pml_widths[1])], axis=-1)
    z = np.max(np.clip(z, a_min=0, a_max=None), axis=-1)
    return ((m + 1) * ln_R * z**m).T


def log_memory(xx, yy, zz, tt, launch_params=((2, 4), (12, 9), 4)):
    """Print analytical memory breakdown and theoretical DRAM traffic."""
    bu, bv, gu, gv, spacing = *launch_params[0], *launch_params[1], launch_params[2]
    pxx = max(xx + 8, (2 * bv + spacing) * gv)
    n   = int(np.ceil((max(yy + 8, 2 * (bu * gu + bv * gv)) - 2 * bu * gu) / (2 * bv * gv)))
    pyy = 2 * bu * gu + n * 2 * bv * gv

    B = 4  # float32 bytes
    fields   = 6 * xx  * yy  * zz * B   # E + H on the physical domain
    buf      = 6 * pxx * pyy * 64 * B   # systolic wavefront buffer (padded)
    cbuf     = 3 * pxx * pyy * 64 * B   # 1/epsilon update coefficients
    output   = 3 * xx  * yy  * zz * B   # one output snapshot (Ex Ey Ez)
    misc     = 2 * pxx * (pyy // 2) * 32 * B  # mask + source layer

    naive_traffic_tb = (2 * tt * fields) / 1024**4  # read + write every step

    print(f"")
    print(f"  Memory breakdown (analytical):")
    print(f"    Padded domain:        {pxx} x {pyy}  (physical: {xx} x {yy})")
    print(f"    E + H fields:         {fields/1024**3:.2f} GB   6 x {xx}x{yy}x{zz} float32")
    print(f"    Systolic buffer:      {buf/1024**3:.2f} GB   wavefront handoff state")
    print(f"    cbuffer (1/epsilon):  {cbuf/1024**3:.2f} GB   update coefficients")
    print(f"    Output snapshot:      {output/1024**3:.2f} GB   Ex Ey Ez at final step")
    print(f"    mask + src:           {misc/1024**3:.3f} GB")
    print(f"    ----------------------------------------")
    print(f"    Total (estimated):    {(fields+buf+cbuf+output+misc)/1024**3:.2f} GB")
    print(f"")
    print(f"  Naive DRAM traffic:     {naive_traffic_tb:.1f} TB  ({tt} steps x read+write 2.5 GB)")
    print(f"  Systolic avoids this by computing multiple steps per diamond pass")
    print(f"")


def simulate(xx, yy, tt, src_omega, src_width, src_delay,
             abs_width, abs_smoothness, pml_widths, use_reduced_precision):
    zz = (128 if use_reduced_precision else 64) - sum(pml_widths)
    epsilon = np.ones((3, xx, yy, zz), np.float32)
    dt = np.float32(0.5)
    source_field = np.zeros((2, 2, xx, yy, 1), np.float32)
    source_field[0, 0, xx // 2, yy // 2, 0] = 1
    source_waveform = np.zeros((tt, 2), np.float32)
    source_waveform[:, 0] = ramped_sin(src_omega, src_width, src_delay, dt, tt)
    source_position = (zz - sum(pml_widths)) // 2 + pml_widths[0]
    abs_mask = absorption_mask(xx, yy, abs_width, abs_smoothness)
    pml_kappa = np.ones((zz, 2), np.float32)
    pml_sigma = pml_sigma_values(pml_widths, zz)
    pml_alpha = np.zeros((zz, 2), np.float32)
    output_steps = (tt - 1, tt, 1)

    print(f"Domain: ({xx}, {yy}, {zz}), steps: {tt}, source_position: {source_position}")
    log_memory(xx, yy, zz, tt)

    stop = threading.Event()
    def _ticker():
        while not stop.wait(timeout=5):
            elapsed = time.time() - t0
            print(f"  running... {elapsed:.0f}s  [{tt:,} steps]", flush=True)
    t0 = time.time()
    threading.Thread(target=_ticker, daemon=True).start()

    out = fdtdz_jax.fdtdz(
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
    )
    out.block_until_ready()
    stop.set()
    elapsed = time.time() - t0
    stats = jax.devices()[0].memory_stats()
    cell_updates = xx * yy * zz * tt
    gcups = cell_updates / elapsed / 1e9
    print(f"  done in {elapsed:.1f}s  [{tt:,} steps]", flush=True)
    print(f"  throughput:           {gcups:.0f}B cell updates/s  (paper quotes 150B)", flush=True)
    print(f"  peak GPU memory:      {stats['peak_bytes_in_use']/1024**3:.2f} GB  (measured)")
    return out


use_reduced_precision = True
src_omega = 0.16

print("=" * 60)
print("Simulation summary")
print("=" * 60)
print(f"  What:       Single Ex dipole source in vacuum")
print(f"  Domain:     1000 x 1000 x 112 cells (dimensionless units,")
print(f"              cell size = 1, speed of light = 1)")
print(f"  Wavelength: 2π / omega = {2 * np.pi / src_omega:.1f} cells")
print(f"  Source:     Ramped continuous-wave sine at domain centre,")
print(f"              z-plane at z=56, gradually switched on via tanh")
print(f"  Boundaries: PML (8 cells) along z  |  adiabatic absorber")
print(f"              (50-cell quadratic conductivity ramp) in x-y")
print(f"  Steps:      10,000  (= {10000 * 0.5:.0f} dimensionless time units)")
print(f"  Output:     Ex field snapshot at final step,")
print(f"              z-slice at z=54 (near centre of domain)")
print(f"  Plot:       Dipole radiation rings propagating in x-y plane")
print("=" * 60)

out = simulate(
    xx=1000,
    yy=1000,
    tt=10000,
    src_omega=0.16,
    src_width=3,
    src_delay=4,
    abs_width=50,
    abs_smoothness=1e-4,
    pml_widths=(8, 8),
    use_reduced_precision=use_reduced_precision,
)

print(f"Output shape: {out.shape}")
print(f"Max |E|: {np.max(np.abs(out)):.6f}")

z_slice = 54 if use_reduced_precision else 22
plt.figure(figsize=(10, 10))
plt.imshow(out[0, 0, :, :, z_slice], vmin=-1e-3, vmax=1e-3, cmap="bwr")
plt.xlabel("y")
plt.ylabel("x")
plt.title("Ex field (z-slice at center)")
plt.colorbar()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
print("Plot saved to output.png")
