"""Device DDPM epsilon-step kernel compared with an independent float reference."""
import numpy as np
import pytest

from opentpu import Config
from opentpu.kernels.ddpm import ddpm_epsilon_step
from opentpu.runtime import Input, Output, launch


@pytest.mark.parametrize('shape', [(1, 2), (3, 17), (17, 33)])
@pytest.mark.parametrize('noise_scale', [0.0, 0.17])
def test_epsilon_step_matches_reference(shape, noise_scale):
    rng = np.random.default_rng(42)
    x, eps, noise = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    coeff = dict(inv_sqrt_alpha_bar=1.4, sqrt_one_minus_alpha_bar=0.7,
                 mean_x0=0.3, mean_xt=0.6, noise_scale=noise_scale, clip_range=1.0)
    got = launch(ddpm_epsilon_step, Config(S=1), sample=Input(x), epsilon=Input(eps),
                 noise=Input(noise), out=Output(shape), M=shape[0], N=shape[1],
                 **coeff).outputs['out']
    x0 = np.clip((x.astype(np.float64) - 0.7 * eps) * 1.4, -1.0, 1.0)
    want = 0.3 * x0 + 0.6 * x + noise_scale * noise
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_clipping_and_final_noise_disabled():
    shape = (1, 4)
    x = np.array([[5.0, -5.0, 0.25, -0.25]], dtype=np.float32)
    eps = np.zeros(shape, dtype=np.float32)
    got = launch(ddpm_epsilon_step, Config(S=1), sample=Input(x), epsilon=Input(eps),
                 noise=Input(np.full(shape, 999.0, np.float32)), out=Output(shape),
                 M=1, N=4, inv_sqrt_alpha_bar=1.0, sqrt_one_minus_alpha_bar=0.0,
                 mean_x0=1.0, mean_xt=0.0, noise_scale=0.0,
                 clip_range=1.0).outputs['out']
    np.testing.assert_array_equal(got, np.clip(x, -1, 1))


def test_ddpm_step_rtl_matches_isa(have_verilator):
    from conftest import assert_same_state
    from opentpu.runtime import compile_kernel, run

    rng = np.random.default_rng(87)
    shape = (3, 17)
    x, eps, noise = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    comp, imgs = compile_kernel(
        ddpm_epsilon_step, Config(S=1, D=32, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16),
        sample=Input(x), epsilon=Input(eps), noise=Input(noise), out=Output(shape),
        M=shape[0], N=shape[1], inv_sqrt_alpha_bar=1.4,
        sqrt_one_minus_alpha_bar=0.7, mean_x0=0.3, mean_xt=0.6,
        noise_scale=0.17, clip_range=1.0)
    assert_same_state(run(comp, [i.copy() for i in imgs], 'isa'),
                      run(comp, [i.copy() for i in imgs], 'rtl'))


@pytest.mark.parametrize('timestep', [0, 10, 90])
def test_step_against_diffusers_fixed_small(timestep):
    torch = pytest.importorskip('torch')
    diffusers = pytest.importorskip('diffusers')
    scheduler = diffusers.DDPMScheduler(num_train_timesteps=100, prediction_type='epsilon',
                                        variance_type='fixed_small', clip_sample=True)
    scheduler.set_timesteps(10)
    shape = (2, 7)
    rng = np.random.default_rng(timestep + 100)
    x, eps = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
    gen_ref = torch.Generator().manual_seed(712)
    gen_noise = torch.Generator().manual_seed(712)
    noise = torch.randn(shape, generator=gen_noise).numpy()
    want = scheduler.step(torch.from_numpy(eps), timestep, torch.from_numpy(x),
                          generator=gen_ref).prev_sample.numpy()
    prev_t = scheduler.previous_timestep(timestep)
    a = scheduler.alphas_cumprod[timestep]
    ap = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    beta = 1 - a
    current_alpha = a / ap
    coeff = dict(inv_sqrt_alpha_bar=float(1 / a.sqrt()),
                 sqrt_one_minus_alpha_bar=float(beta.sqrt()),
                 mean_x0=float(ap.sqrt() * (1 - current_alpha) / beta),
                 mean_xt=float(current_alpha.sqrt() * (1 - ap) / beta),
                 noise_scale=float(scheduler._get_variance(timestep).sqrt()) if timestep > 0 else 0.0,
                 clip_range=float(scheduler.config.clip_sample_range))
    got = launch(ddpm_epsilon_step, Config(S=1), sample=Input(x), epsilon=Input(eps),
                 noise=Input(noise), out=Output(shape), M=shape[0], N=shape[1],
                 **coeff).outputs['out']
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
