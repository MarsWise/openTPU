"""One device-side DDPM update, using precomputed scalar coefficients.

This is a building block, not an integrated scheduler: the caller must supply
the correct coefficients, timestep, and variance noise. No device RNG or loop
is implemented here. All input and output tensors have shape [M, N].
"""
from .. import language as ol
from .eltwise import rows_a_tile


@ol.jit
def ddpm_epsilon_step(sample, epsilon, noise, out, M: int, N: int,
                      inv_sqrt_alpha_bar: float, sqrt_one_minus_alpha_bar: float,
                      mean_x0: float, mean_xt: float, noise_scale: float,
                      clip_range: float):
    """DDPM epsilon prediction with x0 clipping, including supplied noise.

    x0 = clamp((sample - sqrt_one_minus_alpha_bar * epsilon) *
               inv_sqrt_alpha_bar, -clip_range, clip_range)
    out = mean_x0 * x0 + mean_xt * sample + noise_scale * noise

    Constants are compile-time scalar arguments, and `noise_scale` must be
    zero for a deterministic final step. Caller is responsible for validating
    scheduler configuration, coefficient finiteness and shape compatibility.
    """
    R = rows_a_tile(N, 7, reserve_vectors=0)
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        x = ol.load(sample[m0:m0 + n, :])
        eps = ol.load(epsilon[m0:m0 + n, :])
        z = ol.load(noise[m0:m0 + n, :])
        x0 = (x - eps * sqrt_one_minus_alpha_bar) * inv_sqrt_alpha_bar
        x0 = ol.minimum(ol.maximum(x0, -clip_range), clip_range)
        result = x0 * mean_x0 + x * mean_xt + z * noise_scale
        ol.store(out[m0:m0 + n, :], result)


@ol.jit
def ddpm_epsilon_rows(sample, epsilon, noise, y, M: int, N: int,
                      inv_sqrt_alpha_bar: float, sqrt_one_minus_alpha_bar: float,
                      mean_x0: float, mean_xt: float, noise_scale: float, clip_range: float):
    """ddpm_epsilon_step with the result named y (torchdev.Runner.rows' row kernels)."""
    R = rows_a_tile(N, 7, reserve_vectors=0)
    for m0 in range(0, M, R):
        n = min(R, M - m0)
        x = ol.load(sample[m0:m0 + n, :])
        eps = ol.load(epsilon[m0:m0 + n, :])
        z = ol.load(noise[m0:m0 + n, :])
        x0 = (x - eps * sqrt_one_minus_alpha_bar) * inv_sqrt_alpha_bar
        x0 = ol.minimum(ol.maximum(x0, -clip_range), clip_range)
        ol.store(y[m0:m0 + n, :], x0 * mean_x0 + x * mean_xt + z * noise_scale)
