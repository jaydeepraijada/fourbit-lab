"""q4.py: the number formats and the quantized linear layer from Thinking Machines' "End-to-End 4-Bit Training"
(13 Nov 2026), in plain PyTorch so that every step can be read.

Nothing here packs bits. Each function returns ordinary float32 tensors whose values sit exactly on the grid the real
format can store, so the arithmetic is the arithmetic the hardware would do, minus the speed. The packed-bit versions
are the CUDA kernels in kernels.py, and they are checked against these functions.

Formats
    E2M1   the 4-bit float: sign, 2 exponent bits, 1 mantissa bit. Sizes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    E4M3   the 8-bit float: sign, 4 exponent bits, 3 mantissa bits. Largest size 448, smallest 2^-9.
    NVFP4  E2M1 values in blocks of 16, one E4M3 scale per block, one FP32 scale per tensor (or per token).
    MXFP8  E4M3 values in blocks of 32, one power-of-two scale (E8M0) per block.
    S2F6   8-bit fixed point: an int8 divided by 64, so -2 to 127/64, with one power-of-two scale per 32 values.
"""
import math
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

E2M1 = NS(mbits=1, emin=0, emax=2, vmax=6.0)
E4M3 = NS(mbits=3, emin=-6, emax=8, vmax=448.0)


def round_float(v, fmt, mode="rtn"):
    """Round every element of v onto a small float grid.

    mode 'rtn' rounds to the nearest grid point, 'sr' rounds up or down at random with the odds set by the distance
    (stochastic rounding, right on average), 'up' rounds the size up (used for scales that must not be too small).
    """
    a = v.abs()
    _, ex = torch.frexp(a)                                   # a = m * 2^ex with m in [0.5, 1)
    e = (ex - 1).clamp(fmt.emin, fmt.emax)                   # the binade a lives in; below emin it is a subnormal
    step = torch.ldexp(torch.ones_like(a), e - fmt.mbits)    # the gap between neighbours in that binade
    t = a / step
    if mode == "rtn":
        q = torch.round(t)
    elif mode == "sr":
        q = torch.floor(t + torch.rand_like(t))
    else:
        q = torch.ceil(t)
    return torch.copysign((q * step).clamp(max=fmt.vmax), v)


def pow2_scale(amax, target):
    """The smallest power of two s with amax / s <= target. This is what an E8M0 scale can hold."""
    _, ex = torch.frexp(amax / target)
    return torch.ldexp(torch.ones_like(amax), ex.clamp(-127, 127))


def nvfp4(x, mode="rtn", outer="tensor", four_six=False, tile=False, parts=False, outer_scale=None):
    """NVFP4 along the last dimension: blocks of 16, an E4M3 scale per block, an FP32 scale outside.

    outer 'tensor' uses one FP32 scale for everything; 'row' uses one per row (per token for activations).
    four_six tries two scales for each block, one that puts the largest value on 6 and one that puts it on 4, and
    keeps whichever loses less. tile=True shares one scale over each 16 x 16 tile (rows are grouped by 16 as well).
    """
    shp = x.shape
    xb = x.reshape(-1, shp[-1] // 16, 16)
    amax_b = xb.abs().amax(-1, keepdim=True)
    if tile:
        R = xb.shape[0]
        amax_b = amax_b.reshape(R // 16, 16, -1, 1).amax(1, keepdim=True).expand(-1, 16, -1, -1).reshape(R, -1, 1)
    amax_t = xb.abs().amax((1, 2), keepdim=True) if outer == "row" else xb.abs().amax().reshape(1, 1, 1)
    s_outer = (amax_t / ((4.0 if four_six else 6.0) * 448.0)).clamp_min(1e-30)
    if outer_scale is not None:                       # a given FP32 outer scale, as the real kernels use
        s_outer = torch.full_like(s_outer, float(outer_scale))

    def with_target(target):
        sb = round_float(amax_b / target / s_outer, E4M3, "up" if mode == "sr" else "rtn").clamp_min(2.0 ** -9)
        scale = sb * s_outer
        code = round_float(xb / scale, E2M1, mode)
        return code, scale

    code, scale = with_target(6.0)
    if four_six:
        code4, scale4 = with_target(4.0)
        better = ((code4 * scale4 - xb) ** 2).sum(-1, keepdim=True) < ((code * scale - xb) ** 2).sum(-1, keepdim=True)
        code, scale = torch.where(better, code4, code), torch.where(better, scale4, scale)
    if parts:
        return code.reshape(shp), scale.expand_as(xb).reshape(shp)
    return (code * scale).reshape(shp)


def mxfp8(x, mode="rtn"):
    """MXFP8 along the last dimension: blocks of 32, E4M3 values, a power-of-two scale per block."""
    shp = x.shape
    xb = x.reshape(-1, shp[-1] // 32, 32)
    scale = pow2_scale(xb.abs().amax(-1, keepdim=True), 448.0)
    return (round_float(xb / scale, E4M3, mode) * scale).reshape(shp)


def s2f6(x):
    """Block-scaled S2F6 along the last dimension: int8 / 64 with a power-of-two scale per 32 values."""
    shp = x.shape
    xb = x.reshape(-1, shp[-1] // 32, 32)
    scale = pow2_scale(xb.abs().amax(-1, keepdim=True), 127.0 / 64.0)
    return ((torch.round(xb / scale * 64.0).clamp(-128, 127) / 64.0) * scale).reshape(shp)


_RHT = {}


def _rht_parts(device):
    if device not in _RHT:
        H = torch.ones(1, 1)
        while H.shape[0] < 32:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        g = torch.Generator().manual_seed(32)
        signs = (torch.randint(0, 2, (32,), generator=g) * 2 - 1).float()
        _RHT[device] = ((H / math.sqrt(32.0)).to(device), signs.to(device))
    return _RHT[device]


def rht(x, inverse=False):
    """A 32-dimensional random Hadamard transform on each block of 32 along the last dimension.

    Forward: flip signs with a fixed random +-1 vector, then mix with the 32 x 32 Hadamard matrix over sqrt(32).
    The matrix is its own inverse, so the inverse mixes again and then undoes the sign flips.
    """
    H, s = _rht_parts(x.device)
    xb = x.reshape(-1, 32)
    y = (xb @ H) * s if inverse else (xb * s) @ H
    return y.reshape(x.shape)


def combine_quant(y, how):
    """What the expert's output looks like after it has crossed the wire and been decoded."""
    if how == "mxfp8":
        return mxfp8(y)
    if how == "rht_s2f6":
        return rht(s2f6(rht(y)), inverse=True)
    return y


# ------------------------------------------------------------------------------------------- the quantized linear layer
def cfg(fx="none", fw="none", four_six=False, bfmt="exact", bround="sr", keep_w=False, combine="none"):
    """fx, fw: forward formats for activations and weights. bfmt, bround: backward format and rounding."""
    return NS(fx=fx, fw=fw, four_six=four_six, bfmt=bfmt, bround=bround, keep_w=keep_w, combine=combine)


def q_fwd_x(x, c):
    if c.fx == "nvfp4":
        return nvfp4(x, "rtn", outer="row", four_six=c.four_six)       # per-token outer scale
    if c.fx == "mxfp8":
        return mxfp8(x)
    return x


def q_fwd_w(w, c):
    if c.fw == "nvfp4":
        return nvfp4(w, "rtn", outer="tensor")                         # 1 x 16 blocks along the input dimension
    if c.fw == "nvfp4_tile":
        return nvfp4(w, "rtn", outer="tensor", tile=True)              # 16 x 16 tiles
    if c.fw == "mxfp8":
        return mxfp8(w)
    return w


def q_bwd(t, c):
    """Quantize a backward-pass operand. Blocks run along the last dimension, the one the GEMM sums over."""
    if c.bfmt == "mxfp8":
        return mxfp8(t, c.bround)
    if c.bfmt == "nvfp4":
        return nvfp4(t, c.bround, outer="tensor")
    return t


class QLinearFn(torch.autograd.Function):
    """y = x_q w_q^T going forward. Going back, the two gradient GEMMs read re-quantized copies of x_q and w_q.

    x is (M, K): M tokens, K input features. w is (N, K). The forward GEMM sums over K, so both operands are cut into
    blocks along K. DGRAD (dx = g w_q) sums over N and WGRAD (dw = g^T x_q) sums over M, so their operands must be cut
    into blocks along N and along M. That is the "transpose" in the article: same numbers, different neighbours.
    """

    @staticmethod
    def forward(ctx, x, w, c):
        with torch.autocast("cuda", enabled=False):
            xq, wq = q_fwd_x(x, c), q_fwd_w(w, c)
            ctx.save_for_backward(xq, wq)
            ctx.c = c
            return xq @ wq.t()

    @staticmethod
    def backward(ctx, g):
        xq, wq = ctx.saved_tensors
        c = ctx.c
        with torch.autocast("cuda", enabled=False):
            if c.bfmt == "exact":
                return g @ wq, g.t() @ xq, None
            w_t = wq.t() if c.keep_w else q_bwd(wq.t().contiguous(), c)        # (K, N), blocks along N
            dx = q_bwd(g, c) @ w_t.t()                                         # DGRAD
            dw = q_bwd(g.t().contiguous(), c) @ q_bwd(xq.t().contiguous(), c).t()   # WGRAD, blocks along M
            return dx, dw, None


class QLinear(torch.nn.Linear):
    """A bias-free linear layer that follows a cfg(). With the default cfg it is an ordinary linear layer."""

    def __init__(self, n_in, n_out, c=None):
        super().__init__(n_in, n_out, bias=False)
        self.c = c or cfg()

    def forward(self, x):
        c = self.c
        if c.fx == "none" and c.fw == "none" and c.bfmt == "exact":
            return F.linear(x, self.weight)
        x2 = x.reshape(-1, x.shape[-1]).bfloat16().float()                     # activations arrive in BF16
        y = QLinearFn.apply(x2, self.weight.float(), c)
        return y.bfloat16().to(x.dtype).reshape(*x.shape[:-1], -1)             # and leave in BF16


# ---------------------------------------------------------------------------------------------------------------- tests
def selftest(device="cpu"):
    torch.manual_seed(0)
    out = {}
    grid = torch.unique(round_float(torch.linspace(-8, 8, 200001), E2M1).abs())
    out["e2m1_sizes"] = grid.tolist()
    assert grid.tolist() == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    g8 = torch.unique(round_float(torch.cat([torch.linspace(0, 500, 400001), torch.logspace(-12, 0, 20001, base=2)]), E4M3))
    out["e4m3_count_nonneg"], out["e4m3_min_pos"], out["e4m3_max"] = len(g8), g8[1].item(), g8[-1].item()
    assert len(g8) == 127 and g8[1].item() == 2.0 ** -9 and g8[-1].item() == 448.0
    # the article's worked example: a block whose largest value is -54 gets scale 9, and 5.4 becomes 4.5
    blk = torch.tensor([[-54.0, 5.4, 1.0, 20.0, -3.3, 0.7, 9.9, 13.0, 2.0, -27.0, 40.0, 0.2, 6.6, -8.1, 31.0, 4.4]])
    code, scale = nvfp4(blk, parts=True)
    out["article_block"] = dict(values=blk[0].tolist(), scale=scale[0, 0].item(), codes=code[0].tolist(),
                                dequant=(code * scale)[0].tolist())
    assert abs(scale[0, 0].item() - 9.0) < 1e-4 and abs((code * scale)[0, 1].item() - 4.5) < 1e-4
    # stochastic rounding of 2.25 on the FP4 grid: 2 three quarters of the time, 3 one quarter
    r = round_float(torch.full((400000,), 2.25), E2M1, "sr")
    out["sr_2p25"] = dict(frac_2=(r == 2).float().mean().item(), frac_3=(r == 3).float().mean().item(), mean=r.mean().item())
    assert abs(r.mean().item() - 2.25) < 0.01
    # SR is right on average, RTN is not: one vector, many re-quantizations
    x = torch.randn(64, 256, device=device)
    for name, fn in [("nvfp4", lambda m: nvfp4(x, m)), ("mxfp8", lambda m: mxfp8(x, m))]:
        mean_sr = torch.stack([fn("sr") for _ in range(400)]).mean(0)
        out[f"{name}_rel_err_rtn"] = ((fn("rtn") - x).norm() / x.norm()).item()
        out[f"{name}_rel_err_one_sr"] = ((fn("sr") - x).norm() / x.norm()).item()
        out[f"{name}_rel_err_mean_of_400_sr"] = ((mean_sr - x).norm() / x.norm()).item()
    y = torch.randn(8, 128, device=device)
    out["rht_roundtrip_err"] = (rht(rht(y), inverse=True) - y).abs().max().item()
    assert out["rht_roundtrip_err"] < 1e-5
    out["rht_keeps_energy"] = (rht(y).norm() / y.norm()).item()
    # the layer with no quantization gives the same gradients as an ordinary linear layer
    w = torch.randn(64, 32, device=device, requires_grad=True)
    xx = torch.randn(96, 32, device=device, requires_grad=True)
    c0 = cfg(fx="mxfp8", fw="mxfp8", bfmt="exact")
    l1 = (QLinearFn.apply(xx, w, c0) ** 2).sum()
    gx, gw = torch.autograd.grad(l1, [xx, w])
    l2 = ((mxfp8(xx.detach()) @ mxfp8(w.detach()).t()) ** 2).sum()
    out["layer_matches"] = bool(abs(l1.item() - l2.item()) < 1e-3 * abs(l2.item()))
    for b in ["mxfp8", "nvfp4"]:
        for r_ in ["sr", "rtn"]:
            c1 = cfg(fx="nvfp4", fw="nvfp4", four_six=True, bfmt=b, bround=r_)
            lq = (QLinearFn.apply(xx, w, c1) ** 2).sum()
            gxq, gwq = torch.autograd.grad(lq, [xx, w])
            cref = cfg(fx="nvfp4", fw="nvfp4", four_six=True, bfmt="exact")
            gxr, gwr = torch.autograd.grad((QLinearFn.apply(xx, w, cref) ** 2).sum(), [xx, w])
            out[f"grad_rel_err_{b}_{r_}"] = dict(dx=((gxq - gxr).norm() / gxr.norm()).item(), dw=((gwq - gwr).norm() / gwr.norm()).item())
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=1))
