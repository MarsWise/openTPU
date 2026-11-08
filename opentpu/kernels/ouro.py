"""Device primitives for Ouro; not a complete recurrent model executor."""
from .. import language as ol
from .lib import rmsnorm
from .mlp import swiglu_down


def attention_input(x, gamma, eps):
    """Normalize the full residual before the separate attention operator."""
    if gamma.shape != (x.cols,):
        raise ol.CompileError('Ouro attention input norm dimensions do not match')
    return rmsnorm(x, gamma, eps)


def attention_qkv(x, gamma, wq, wk, wv, eps):
    """Pre-attention RMSNorm and Q/K/V projections (without RoPE or KV update)."""
    if ol.num_programs() != 1:
        raise ol.CompileError('Ouro QKV requires a full hidden vector on one slice')
    if any(w.shape[1] != x.cols for w in (wq, wk, wv)):
        raise ol.CompileError('Ouro QKV projection dimensions do not match')
    if wk.shape[0] != wv.shape[0]:
        raise ol.CompileError('Ouro K and V projection widths must match')
    a = ol.quantize(attention_input(x, gamma, eps))
    return ol.dot(a, wq), ol.dot(a, wk), ol.dot(a, wv)


def attention_residual(x, attention_output, gamma, eps):
    """Residual + post-attention RMSNorm, after the full output projection."""
    if ol.num_programs() != 1:
        raise ol.CompileError('Ouro post-attention norm requires a full hidden vector on one slice')
    if attention_output.shape != x.shape or gamma.shape != (x.cols,):
        raise ol.CompileError('Ouro attention residual dimensions do not match')
    return x + rmsnorm(attention_output, gamma, eps)


def sandwich_mlp(x, gamma_before, gamma_after, gate, up, down, eps, chunk=None):
    """Ouro residual + RMSNorm_after(MLP(RMSNorm_before(residual))).

    The model applies both MLP norms before the residual addition. This
    primitive is intentionally limited to one slice: post-MLP RMSNorm needs
    the entire hidden vector, so sharded execution must gather it first.
    """
    if ol.num_programs() != 1:
        raise ol.CompileError('Ouro sandwich MLP requires a full hidden vector on one slice')
    if x.cols != down.shape[0] or gamma_before.shape != (x.cols,) or gamma_after.shape != (x.cols,):
        raise ol.CompileError('Ouro sandwich MLP hidden dimensions do not match')
    stationary = ol.quantize(rmsnorm(x, gamma_before, eps))
    mlp_out = swiglu_down(stationary, gate, up, down, chunk=chunk)
    return x + rmsnorm(mlp_out, gamma_after, eps)
