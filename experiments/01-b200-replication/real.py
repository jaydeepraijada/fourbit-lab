"""real.py: 4-bit and 8-bit multiplies on real Blackwell tensor cores, inside the PyTorch process.

    python3 mo.py modal deploy real.py
    FOURBIT_APP=fourbit-real python3 mo.py call probe

fb.cu is compiled in the container and loaded with ctypes. FB wraps it: plans, GEMMs on torch tensors, quantize kernels.
"""
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("fourbit-real")
vol = modal.Volume.from_name("fourbit-data", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12").entrypoint([])
         .pip_install("torch==2.8.0", "numpy==2.3.5", "tiktoken==0.9.0")
         .env({"PYTHONUNBUFFERED": "1"})
         .add_local_python_source("q4", "train")
         .add_local_file(HERE / "fb.cu", "/root/fb.cu"))

BF16, FP8, NVFP4 = 0, 1, 2


def build_lib():
    so = "/tmp/libfb.so"
    if not os.path.exists(so):
        r = subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_100a", "-shared", "-Xcompiler", "-fPIC", "/root/fb.cu", "-o", so,
                            "-lcublasLt"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed:\n" + r.stderr[-3000:])
    return so


class FB:
    """cuBLASLt GEMMs and our kernels on torch tensors. All matrices are torch row-major tensors."""

    def __init__(self):
        import torch
        self.torch = torch
        L = ctypes.CDLL(build_lib(), mode=os.RTLD_NOW | os.RTLD_DEEPBIND)
        vp, cl, ci, cf, cs = ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_float, ctypes.c_size_t
        L.fb_plan.restype = vp
        L.fb_plan.argtypes = [ci, ci, ci, cl, cl, cl, cl, cl, cl, ci, cs, vp, vp, ctypes.POINTER(ci)]
        L.fb_run.argtypes = [vp, cf, vp, vp, vp, vp, vp, vp, cs]
        L.fb_pick.argtypes = [vp, ci]
        L.fb_nalgo.argtypes = [vp]
        L.fb_quant_nvfp4.argtypes = [vp, ci, cl, cl, cf, vp, vp, vp, cf, ci, ctypes.c_uint]
        L.fb_quant_nvfp4.restype = None
        L.fb_quant_fp8.argtypes = [vp, vp, cl, cf, vp, ci, ci, ctypes.c_uint]
        L.fb_quant_fp8.restype = None
        L.fb_quant_fp8_f32.argtypes = [vp, cl, cf, vp, ci, ctypes.c_uint]
        L.fb_quant_fp8_f32.restype = None
        L.fb_set_hw.argtypes = [ci]
        L.fb_set_hw.restype = None
        L.fb_set_copy_sr.argtypes = [ci]
        L.fb_set_copy_sr.restype = None
        self.L = L
        torch.zeros(1, device="cuda")                      # make sure torch owns the context first
        st = L.fb_init()
        assert st == 0, f"cublasLtCreate {st}"
        self.wsz = 256 << 20
        self.ws = torch.empty(self.wsz, dtype=torch.uint8, device="cuda")
        self.one = torch.ones(1, device="cuda")
        self.plans = {}
        self.seed = 12345

    # ---------------------------------------------------------------- GEMM
    def plan(self, kind, tA, tB, a, b, c, out_f32, sA=None, sB=None):
        key = (kind, tA, tB, a, b, c, out_f32)
        if key not in self.plans:
            st = ctypes.c_int(0)
            p = self.L.fb_plan(kind, tA, tB, a[0], a[1], b[0], b[1], c[0], c[1], 1 if out_f32 else 0, self.wsz,
                               sA if sA is not None else self.one.data_ptr(), sB if sB is not None else self.one.data_ptr(), ctypes.byref(st))
            if not p or st.value != 0:
                raise RuntimeError(f"fb_plan failed kind={kind} ops=({tA},{tB}) a={a} b={b} c={c}: status {st.value}")
            self.plans[key] = [p, False]
        return self.plans[key]

    def run(self, pl, alpha, A, B, C, sA=None, sB=None):
        st = self.L.fb_run(pl[0], float(alpha), A.data_ptr(), B.data_ptr(), C.data_ptr(), sA.data_ptr() if sA is not None else None,
                           sB.data_ptr() if sB is not None else None, self.ws.data_ptr(), self.wsz)
        if st != 0:
            raise RuntimeError(f"cublasLtMatmul status {st}")

    def tune(self, pl, *args, **kw):
        """Time every algorithm cuBLASLt offers for this plan once and keep the fastest."""
        if pl[1]:
            return
        torch = self.torch
        best, bi = 1e9, None
        for i in range(self.L.fb_nalgo(pl[0])):
            if self.L.fb_pick(pl[0], i) != 0:
                continue
            try:
                self.run(pl, *args, **kw); torch.cuda.synchronize()
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); self.run(pl, *args, **kw); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                if min(ts) < best:
                    best, bi = min(ts), i
            except RuntimeError:
                continue
        assert bi is not None, "no working algorithm"
        self.L.fb_pick(pl[0], bi)
        pl[1] = True

    def gemm(self, kind, tA, tB, A, a_shape, B, b_shape, c_shape, alpha=1.0, sA=None, sB=None, out_f32=False, out=None):
        """C = alpha op(A) op(B) in column-major terms. Shapes are (rows, cols) of the stored column-major matrices.
        A torch tensor (R, K) row-major is the column-major matrix (K, R)."""
        torch = self.torch
        if out is None:
            out = torch.empty(c_shape[1], c_shape[0], dtype=torch.float32 if out_f32 else torch.bfloat16, device="cuda")
        pl = self.plan(kind, tA, tB, a_shape, b_shape, c_shape, out_f32, sA.data_ptr() if sA is not None else None,
                       sB.data_ptr() if sB is not None else None)
        if not pl[1]:
            self.tune(pl, alpha, A, B, out, sA, sB)
        self.run(pl, alpha, A, B, out, sA, sB)
        return out

    # ---------------------------------------------------------------- quantizers
    def nvfp4(self, x, outer, f8_scale=None, gelu=False):
        """x (rows, K) BF16 or FP32 -> dict(codes, scales, outer[, f8, s8]). Blocks of 16 along K, cuBLASLt scale layout."""
        torch = self.torch
        rows, K = x.shape
        assert K % 64 == 0
        codes = torch.empty(rows * K // 16, dtype=torch.int64, device="cuda")
        scales = torch.zeros(((rows + 127) // 128) * (K // 64) * 512, dtype=torch.uint8, device="cuda")
        f8 = torch.empty(rows, K, dtype=torch.uint8, device="cuda") if f8_scale is not None else None
        self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
        self.L.fb_quant_nvfp4(x.data_ptr(), 1 if x.dtype == torch.bfloat16 else 0, rows, K, float(outer), codes.data_ptr(),
                              scales.data_ptr(), f8.data_ptr() if f8 is not None else None, float(f8_scale or 1.0), 1 if gelu else 0, self.seed)
        return dict(codes=codes, scales=scales, outer=float(outer), f8=f8, s8=f8_scale, shape=(rows, K))

    def fp8(self, x, scale, sr=True, gelu=False, grad_of=None):
        """x BF16 (or FP32 without options) -> FP8 E4M3 bytes with one tensor scale."""
        torch = self.torch
        out = torch.empty(x.shape, dtype=torch.uint8, device="cuda")
        self.seed = (self.seed * 1103515245 + 12345) & 0x7FFFFFFF
        if x.dtype == torch.bfloat16:
            self.L.fb_quant_fp8(x.data_ptr(), grad_of.data_ptr() if grad_of is not None else None, x.numel(), float(scale), out.data_ptr(),
                                1 if sr else 0, 1 if gelu else 0, self.seed)
        else:
            assert not gelu and grad_of is None
            self.L.fb_quant_fp8_f32(x.data_ptr(), x.numel(), float(scale), out.data_ptr(), 1 if sr else 0, self.seed)
        return out


def e4m3_decode_table(device):
    import torch
    v = torch.zeros(256)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        x = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        v[c] = -x if s else x
    return v.to(device)


@app.function(image=image, gpu="B200", cpu=8.0, memory=65536, timeout=2400, volumes={"/vol": vol})
def probe(widths: str = "2048,4096,8192", M: int = 32768):
    import torch
    import torch.nn.functional as F
    import q4
    out = dict(gpu=torch.cuda.get_device_name(0))
    hdr = subprocess.run(r"""H=/usr/local/cuda/include
grep -n -E '^__CUDA_HOSTDEVICE_FP[48]_DECL__|__nv_cvt_[a-z0-9_]*\(' $H/cuda_fp4.h $H/cuda_fp8.h | cut -c1-170 | head -60
grep -n -E 'MATRIX_SCALE_|cudaRoundMode|__NV_(SAT|NOSAT|E)' $H/cublasLt.h $H/cuda_fp8.h $H/cuda_fp4.h | cut -c1-170 | head -30
grep -rn -E 'enum cudaRoundMode' -A 8 $H/driver_types.h | head -14""", shell=True, capture_output=True, text=True).stdout
    out["headers"] = hdr
    print(hdr, flush=True)
    fb = FB()
    dec8 = e4m3_decode_table("cuda")
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
    sync = torch.cuda.synchronize

    def timeit(fn, reps=15, warm=3):
        for _ in range(warm):
            fn()
        sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
        ts.sort()
        return 1000 * ts[len(ts) // 2]

    # ------------------------------------------------------------ correctness, small shapes
    torch.manual_seed(0)
    Mt, K, N = 4096, 1024, 2048
    x = (torch.randn(Mt, K, device="cuda") * torch.exp(torch.randn(Mt, 1, device="cuda"))).bfloat16()
    w = torch.randn(N, K, device="cuda") * 0.03
    ox, ow = x.float().abs().max().item() / (6 * 448), w.abs().max().item() / (6 * 448)
    s8x = 1.125 * x.float().abs().max().item() / 448
    qx, qw = fb.nvfp4(x, ox, f8_scale=s8x), fb.nvfp4(w, ow)
    sim_x, sim_w = q4.nvfp4(x.float()), q4.nvfp4(w)
    y_hw = fb.gemm(NVFP4, 1, 0, qw["codes"], (K, N), qx["codes"], (K, Mt), (N, Mt), alpha=ox * ow, sA=qw["scales"], sB=qx["scales"], out_f32=True)
    y_sim = sim_x.double() @ sim_w.double().t()
    y_ref = x.double() @ w.double().t()
    c = dict(nvfp4_hw_vs_simulation=rel(y_hw, y_sim), nvfp4_hw_vs_unquantized=rel(y_hw, y_ref), simulation_vs_unquantized=rel(y_sim, y_ref))
    xq8 = dec8[qx["f8"].long()] * s8x
    c["fp8_copy_of_xq_rel_err"] = rel(xq8, sim_x)
    c["fp8_copy_mean_signed_err"] = ((xq8 - sim_x).mean() / sim_x.abs().mean()).item()
    out["correct_nvfp4"] = c
    print("correct_nvfp4", c, flush=True)
    # FP8 with scalar scales: the three operand patterns the backward pass needs
    g = (torch.randn(Mt, N, device="cuda") * 1e-4).bfloat16()
    sg, sw = g.float().abs().max().item() / 448, w.abs().max().item() / 448
    g8, w8 = fb.fp8(g, sg, sr=False), fb.fp8(w, sw, sr=False)
    gd, wd, xd = dec8[g8.long()] * sg, dec8[w8.long()] * sw, xq8
    S = lambda v: torch.tensor([v], device="cuda")
    f = {}
    try:   # DGRAD: dx (Mt, K) = g (Mt, N) @ w (N, K);  column-major: (K, Mt) = Wc (K, N) . Gc (N, Mt)
        dx = fb.gemm(FP8, 0, 0, w8, (K, N), g8, (N, Mt), (K, Mt), sA=S(sw), sB=S(sg), out_f32=True)
        f["dgrad_NN"] = rel(dx, gd.double() @ wd.double())
    except Exception as e:
        f["dgrad_NN"] = "FAILED " + str(e)[:200]
    try:   # WGRAD: dw (N, K) = g^T x;  column-major: (K, N) = Xc (K, Mt) . Gc^T (Mt, N)
        dw = fb.gemm(FP8, 0, 1, qx["f8"], (K, Mt), g8, (N, Mt), (K, N), sA=S(s8x), sB=S(sg), out_f32=True)
        f["wgrad_NT"] = rel(dw, gd.double().t() @ xd.double())
    except Exception as e:
        f["wgrad_NT"] = "FAILED " + str(e)[:200]
    try:   # FPROP in FP8: (N, Mt) = Wc^T . Xc
        y8 = fb.gemm(FP8, 1, 0, w8, (K, N), qx["f8"], (K, Mt), (N, Mt), sA=S(sw), sB=S(s8x), out_f32=True)
        f["fprop_TN"] = rel(y8, xd.double() @ wd.double().t())
    except Exception as e:
        f["fprop_TN"] = "FAILED " + str(e)[:200]
    out["correct_fp8"] = f
    print("correct_fp8", f, flush=True)
    del x, w, g, qx, qw, y_hw, y_sim, y_ref

    # ------------------------------------------------------------ timings at realistic widths
    T = {}
    for d in [int(v) for v in widths.split(",")]:
        D4 = 4 * d
        x = torch.randn(M, d, device="cuda").bfloat16()
        w1 = (torch.randn(D4, d, device="cuda") * 0.02)
        w2 = (torch.randn(d, D4, device="cuda") * 0.02)
        w1b, w2b = w1.bfloat16(), w2.bfloat16()
        g = (torch.randn(M, d, device="cuda") * 1e-4).bfloat16()
        r = dict(M=M, d=d)
        with torch.no_grad():
            y1 = x @ w1b.t()
            x2 = F.gelu(y1)
            # BF16, as PyTorch does it
            r["bf16_fprop1"] = timeit(lambda: x @ w1b.t())
            r["bf16_gelu"] = timeit(lambda: F.gelu(y1))
            r["bf16_fprop2"] = timeit(lambda: x2 @ w2b.t())
            r["bf16_dgrad2"] = timeit(lambda: g @ w2b)
            dx2 = g @ w2b
            r["bf16_wgrad2"] = timeit(lambda: g.t() @ x2)
            r["bf16_gelu_bwd"] = timeit(lambda: dx2 * (y1 > 0))          # a lower bound: PyTorch's GELU backward costs more
            g1 = dx2
            r["bf16_dgrad1"] = timeit(lambda: g1 @ w1b)
            r["bf16_wgrad1"] = timeit(lambda: g1.t() @ x)
            r["amax_pass_M_4d"] = timeit(lambda: torch.aminmax(y1))
            # NVFP4 forward, real
            ox, ow1, ow2 = 4.0 / (6 * 448), w1.abs().max().item() / (6 * 448), w2.abs().max().item() / (6 * 448)
            ox2 = 8.0 / (6 * 448)
            r["q_x_nvfp4"] = timeit(lambda: fb.nvfp4(x, ox))
            r["q_x_nvfp4_with_fp8_copy"] = timeit(lambda: fb.nvfp4(x, ox, f8_scale=0.01))
            r["q_w1_nvfp4"] = timeit(lambda: fb.nvfp4(w1, ow1))
            qx, qw1, qw2 = fb.nvfp4(x, ox, f8_scale=0.012), fb.nvfp4(w1, ow1), fb.nvfp4(w2, ow2)
            y1h = torch.empty(M, D4, dtype=torch.bfloat16, device="cuda")
            r["nvfp4_fprop1"] = timeit(lambda: fb.gemm(NVFP4, 1, 0, qw1["codes"], (d, D4), qx["codes"], (d, M), (D4, M), alpha=ox * ow1, sA=qw1["scales"], sB=qx["scales"], out=y1h))
            r["q_gelu_x2_nvfp4"] = timeit(lambda: fb.nvfp4(y1, ox2, gelu=True))
            r["q_gelu_x2_nvfp4_with_fp8_copy"] = timeit(lambda: fb.nvfp4(y1, ox2, f8_scale=0.02, gelu=True))
            qx2 = fb.nvfp4(y1, ox2, f8_scale=0.02, gelu=True)
            yh = torch.empty(M, d, dtype=torch.bfloat16, device="cuda")
            r["nvfp4_fprop2"] = timeit(lambda: fb.gemm(NVFP4, 1, 0, qw2["codes"], (D4, d), qx2["codes"], (D4, M), (d, M), alpha=ox2 * ow2, sA=qw2["scales"], sB=qx2["scales"], out=yh))
            # FP8 pieces, real
            S = lambda v: torch.tensor([v], device="cuda")
            s1 = S(1.0)
            r["q_g_fp8_sr"] = timeit(lambda: fb.fp8(g, 1e-6, sr=True))
            r["q_gelu_fp8_M_4d"] = timeit(lambda: fb.fp8(y1, 0.02, sr=False, gelu=True))
            r["q_gelubwd_fp8_sr_M_4d"] = timeit(lambda: fb.fp8(dx2, 1e-6, sr=True, grad_of=y1))
            g8, g18 = fb.fp8(g, 1e-6), fb.fp8(dx2, 1e-6, grad_of=y1)
            w18, w28, x8, x28 = fb.fp8(w1, 0.001), fb.fp8(w2, 0.001), qx["f8"], qx2["f8"]
            try:
                o1 = torch.empty(M, D4, dtype=torch.bfloat16, device="cuda")
                r["fp8_fprop1"] = timeit(lambda: fb.gemm(FP8, 1, 0, w18, (d, D4), x8, (d, M), (D4, M), sA=s1, sB=s1, out=o1))
                o2 = torch.empty(M, d, dtype=torch.bfloat16, device="cuda")
                r["fp8_fprop2"] = timeit(lambda: fb.gemm(FP8, 1, 0, w28, (D4, d), x28, (D4, M), (d, M), sA=s1, sB=s1, out=o2))
                r["fp8_dgrad2"] = timeit(lambda: fb.gemm(FP8, 0, 0, w28, (D4, d), g8, (d, M), (D4, M), sA=s1, sB=s1, out=o1))
                o3 = torch.empty(d, D4, dtype=torch.float32, device="cuda")
                r["fp8_wgrad2"] = timeit(lambda: fb.gemm(FP8, 0, 1, x28, (D4, M), g8, (d, M), (D4, d), sA=s1, sB=s1, out_f32=True, out=o3))
                r["fp8_dgrad1"] = timeit(lambda: fb.gemm(FP8, 0, 0, w18, (d, D4), g18, (D4, M), (d, M), sA=s1, sB=s1, out=o2))
                o4 = torch.empty(D4, d, dtype=torch.float32, device="cuda")
                r["fp8_wgrad1"] = timeit(lambda: fb.gemm(FP8, 0, 1, x8, (d, M), g18, (D4, M), (d, D4), sA=s1, sB=s1, out_f32=True, out=o4))
            except Exception as e:
                r["fp8_error"] = str(e)[:300]
        r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        T[str(d)] = r
        print(d, json.dumps(r), flush=True)
        del x, w1, w2, w1b, w2b, g, y1, x2, dx2, qx, qw1, qw2, qx2, y1h, yh
        torch.cuda.empty_cache()
    out["timings_ms"] = T
    os.makedirs("/vol/real", exist_ok=True)
    json.dump(out, open("/vol/real/probe.json", "w"), indent=1)
    vol.commit()
    return out


@app.function(image=image, gpu="B200", cpu=8.0, memory=65536, timeout=1800, volumes={"/vol": vol})
def kprobe(d: int = 8192, M: int = 32768):
    """Version-2 kernels (hardware conversion instructions) against the software kernels: same answers? how much faster?"""
    import math
    import torch
    import q4
    fb = FB()
    dec8 = e4m3_decode_table("cuda")
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
    out = dict(gpu=torch.cuda.get_device_name(0))

    def timeit(fn, reps=15, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        ts.sort()
        return round(1000 * ts[len(ts) // 2], 4)

    torch.manual_seed(0)
    Mt, K, N = 4096, 1024, 2048
    x = (torch.randn(Mt, K, device="cuda") * torch.exp(torch.randn(Mt, 1, device="cuda"))).bfloat16()
    w = torch.randn(N, K, device="cuda") * 0.03
    ox, ow = x.float().abs().max().item() / (6 * 448), w.abs().max().item() / (6 * 448)
    s8 = 1.125 * x.float().abs().max().item() / 448
    sim_x, sim_w = q4.nvfp4(x.float()), q4.nvfp4(w)
    y_sim = sim_x.double() @ sim_w.double().t()
    res = {}
    for hw in (0, 1):
        fb.L.fb_set_hw(hw)
        qx, qw = fb.nvfp4(x, ox, f8_scale=s8), fb.nvfp4(w, ow)
        res[hw] = (qx, qw)
        y = fb.gemm(NVFP4, 1, 0, qw["codes"], (K, N), qx["codes"], (K, Mt), (N, Mt), alpha=ox * ow, sA=qw["scales"], sB=qx["scales"], out_f32=True)
        c8 = dec8[qx["f8"].long()] * s8
        out[f"nvfp4_hw{hw}"] = dict(gemm_vs_simulation=rel(y, y_sim), fp8_copy_rel_err=rel(c8, sim_x), fp8_copy_mean_signed=((c8 - sim_x).mean() / sim_x.abs().mean()).item())
    out["nvfp4_codes_identical"] = (res[0][0]["codes"] == res[1][0]["codes"]).float().mean().item()
    out["nvfp4_weight_codes_identical"] = (res[0][1]["codes"] == res[1][1]["codes"]).float().mean().item()
    out["nvfp4_scales_identical"] = (res[0][0]["scales"] == res[1][0]["scales"]).float().mean().item()
    # FP8: nearest must agree between the two kernels; stochastic must be right on average
    g = (torch.randn(Mt, N, device="cuda") * 1e-4 * torch.exp(torch.randn(Mt, 1, device="cuda"))).bfloat16()
    y1 = (torch.randn(Mt, N, device="cuda") * 1.5).bfloat16()
    sg = g.float().abs().max().item() / 448
    gf, yf = g.float(), y1.float()
    gelu_ref = torch.nn.functional.gelu(yf)
    ggrad_ref = gf * (0.5 * (1 + torch.erf(yf / math.sqrt(2))) + yf * torch.exp(-0.5 * yf * yf) / math.sqrt(2 * math.pi))
    sy = 1.1 * gelu_ref.abs().max().item() / 448
    sgg = 1.2 * ggrad_ref.abs().max().item() / 448
    f = {}
    for hw in (0, 1):
        fb.L.fb_set_hw(hw)
        a = fb.fp8(g, sg, sr=False)
        f[f"rtn_hw{hw}"] = a
        out[f"fp8_hw{hw}"] = dict(
            rtn_rel_err=rel(dec8[a.long()] * sg, gf),
            gelu_rtn_rel_err=rel(dec8[fb.fp8(y1, sy, sr=False, gelu=True).long()] * sy, gelu_ref),
            one_sr_rel_err=rel(dec8[fb.fp8(g, sg, sr=True).long()] * sg, gf))
        acc = torch.zeros_like(gf); acc2 = torch.zeros_like(gf)
        for _ in range(64):
            acc += dec8[fb.fp8(g, sg, sr=True).long()] * sg
            acc2 += dec8[fb.fp8(g, sgg, sr=True, grad_of=y1).long()] * sgg
        out[f"fp8_hw{hw}"].update(mean_of_64_sr_rel_err=rel(acc / 64, gf), mean_signed_sr=((acc / 64 - gf).mean() / gf.abs().mean()).item(),
                                  gelu_bwd_mean_of_64_sr_rel_err=rel(acc2 / 64, ggrad_ref))
    out["fp8_rtn_codes_identical"] = (f["rtn_hw0"] == f["rtn_hw1"]).float().mean().item()
    print(json.dumps(out, indent=1), flush=True)
    del x, w, g, y1, res, f

    # timings at one realistic width
    D4 = 4 * d
    x = torch.randn(M, d, device="cuda").bfloat16()
    y1 = torch.randn(M, D4, device="cuda").bfloat16()
    dx2 = (torch.randn(M, D4, device="cuda") * 1e-4).bfloat16()
    w1 = torch.randn(D4, d, device="cuda") * 0.02
    T = {}
    for hw in (0, 1):
        fb.L.fb_set_hw(hw)
        T[f"hw{hw}"] = dict(
            q_x_nvfp4=timeit(lambda: fb.nvfp4(x, 0.002)), q_x_nvfp4_fp8copy=timeit(lambda: fb.nvfp4(x, 0.002, f8_scale=0.012)),
            q_gelu_x2_nvfp4=timeit(lambda: fb.nvfp4(y1, 0.002, gelu=True)), q_gelu_x2_nvfp4_fp8copy=timeit(lambda: fb.nvfp4(y1, 0.002, f8_scale=0.012, gelu=True)),
            q_w1_nvfp4_fp8copy=timeit(lambda: fb.nvfp4(w1, 1e-4, f8_scale=1e-3)),
            q_x_fp8_rtn=timeit(lambda: fb.fp8(x, 0.012, sr=False)), q_x_fp8_sr=timeit(lambda: fb.fp8(x, 0.012, sr=True)),
            q_gelu_fp8_rtn=timeit(lambda: fb.fp8(y1, 0.012, sr=False, gelu=True)), q_gelubwd_fp8_sr=timeit(lambda: fb.fp8(dx2, 1e-6, sr=True, grad_of=y1)))
    T["torch_gelu_M_4d"] = timeit(lambda: torch.nn.functional.gelu(y1))
    T["torch_copy_M_4d"] = timeit(lambda: y1.clone())
    T["values_M_d"], T["values_M_4d"] = M * d, M * D4
    out["timings_ms"] = T
    print(json.dumps(T, indent=1), flush=True)
    json.dump(out, open("/vol/real/kprobe.json", "w"), indent=1)
    vol.commit()
    return out


# ====================================================================================================== real training
CFG = dict(n_layer=4, n_head=16, block=256, vocab=8192)


def load_compact(split, device, vocab):
    """TinyStories tokens remapped to the `vocab - 1` most frequent GPT-2 tokens plus one catch-all id."""
    import numpy as np
    import torch
    tr = np.fromfile("/vol/data/train.bin", dtype=np.uint16)
    counts = np.bincount(tr, minlength=65536)
    top = np.argsort(-counts)[:vocab - 1]
    lut = np.full(65536, vocab - 1, dtype=np.int16)
    lut[top] = np.arange(vocab - 1, dtype=np.int16)
    cover = counts[top].sum() / counts.sum()
    arr = tr if split == "train" else np.fromfile(f"/vol/data/{split}.bin", dtype=np.uint16)
    return torch.from_numpy(lut[arr]).to(device), top, float(cover)


def build_real_model(arm, d, fb, shared, cfg=CFG):
    """arm: 'bf16' (PyTorch autocast), 'fp8' (real 8-bit GEMMs both ways), 'nvfp4' (real 4-bit forward, 8-bit backward)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import q4
    L, H, T, V = cfg["n_layer"], cfg["n_head"], cfg["block"], cfg["vocab"]
    D4 = 4 * d

    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w1, w2, mod):
            B_, T_, _ = x.shape
            M = B_ * T_
            x2 = x.reshape(M, d)
            x2 = (x2 if x2.dtype == torch.bfloat16 else x2.bfloat16()).contiguous()
            ax, aw1, aw2 = mod.amax("x", x2, 2.0), mod.amax("w1", w1, 1.25), mod.amax("w2", w2, 1.25)
            sx, sw1, sw2 = ax / 448, aw1 / 448, aw2 / 448
            if mod.kind == "nvfp4":
                ox, ow1, ow2 = ax / 2688, aw1 / 2688, aw2 / 2688
                qx, qw1 = fb.nvfp4(x2, ox, f8_scale=sx), fb.nvfp4(w1, ow1, f8_scale=sw1)
                y1 = fb.gemm(NVFP4, 1, 0, qw1["codes"], (d, D4), qx["codes"], (d, M), (D4, M), alpha=ox * ow1, sA=qw1["scales"], sB=qx["scales"])
                ay = mod.amax("y1", y1, 2.0)
                sx2, ox2 = ay / 448, ay / 2688
                qx2, qw2 = fb.nvfp4(y1, ox2, f8_scale=sx2, gelu=True), fb.nvfp4(w2, ow2, f8_scale=sw2)
                y = fb.gemm(NVFP4, 1, 0, qw2["codes"], (D4, d), qx2["codes"], (D4, M), (d, M), alpha=ox2 * ow2, sA=qw2["scales"], sB=qx2["scales"])
                x8, x28, w18, w28 = qx["f8"], qx2["f8"], qw1["f8"], qw2["f8"]
                if mod.exact:                    # dequantized forward tensors for a high-precision reference backward
                    ctx.dq = (q4.nvfp4(x2.float(), outer_scale=ox), q4.nvfp4(w1.float(), outer_scale=ow1),
                              q4.nvfp4(F.gelu(y1.float()), outer_scale=ox2), q4.nvfp4(w2.float(), outer_scale=ow2))
            else:
                x8, w18 = fb.fp8(x2, sx, sr=False), fb.fp8(w1, sw1, sr=False)
                y1 = fb.gemm(FP8, 1, 0, w18, (d, D4), x8, (d, M), (D4, M), sA=mod.S("w1", sw1), sB=mod.S("x", sx))
                ay = mod.amax("y1", y1, 2.0)
                sx2 = ay / 448
                x28, w28 = fb.fp8(y1, sx2, sr=False, gelu=True), fb.fp8(w2, sw2, sr=False)
                y = fb.gemm(FP8, 1, 0, w28, (D4, d), x28, (D4, M), (d, M), sA=mod.S("w2", sw2), sB=mod.S("x2", sx2))
            ctx.st = (mod, M, y1, x8, sx, x28, sx2, w18, sw1, w28, sw2, x.shape, x.dtype)
            if mod.cap is not None:
                mod.cap.update(x=x2, y1=y1, y=y)
            return y.view(B_, T_, d).to(x.dtype)

        @staticmethod
        def backward(ctx, g):
            mod, M, y1, x8, sx, x28, sx2, w18, sw1, w28, sw2, shp, dt = ctx.st
            g2 = g.reshape(M, d)
            g2 = (g2 if g2.dtype == torch.bfloat16 else g2.bfloat16()).contiguous()
            if mod.cap is not None:
                mod.cap["g"] = g2
            if mod.exact:
                xq, w1q, x2q, w2q = ctx.dq
                gf, yf = g2.float(), y1.float()
                dw2, dx2 = gf.t() @ x2q, gf @ w2q
                g1 = dx2 * (0.5 * (1 + torch.erf(yf * 0.7071067811865476)) + yf * torch.exp(-0.5 * yf * yf) * 0.3989422804014327)
                return (g1 @ w1q).view(shp).to(dt), g1.t() @ xq, dw2, None
            sg = mod.amax("g", g2, 4.0) / 448
            g8 = fb.fp8(g2, sg, sr=mod.sr)
            Sg = mod.S("g", sg)
            dw2 = fb.gemm(FP8, 0, 1, x28, (D4, M), g8, (d, M), (D4, d), sA=mod.S("x2", sx2), sB=Sg, out_f32=True)       # WGRAD
            dx2 = fb.gemm(FP8, 0, 0, w28, (D4, d), g8, (d, M), (D4, M), sA=mod.S("w2", sw2), sB=Sg)                     # DGRAD
            sg1 = 1.13 * mod.amax("dx2", dx2, 4.0) / 448
            g18 = fb.fp8(dx2, sg1, sr=mod.sr, grad_of=y1)                                                                  # times GELU'(y1)
            Sg1 = mod.S("g1", sg1)
            dw1 = fb.gemm(FP8, 0, 1, x8, (d, M), g18, (D4, M), (d, D4), sA=mod.S("x", sx), sB=Sg1, out_f32=True)        # WGRAD
            dx = fb.gemm(FP8, 0, 0, w18, (d, D4), g18, (D4, M), (d, M), sA=mod.S("w1", sw1), sB=Sg1)                    # DGRAD
            return dx.view(shp).to(dt), dw1, dw2, None

    class MLP(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.w1 = nn.Parameter(torch.randn(D4, d) * 0.02)
            self.w2 = nn.Parameter(torch.randn(d, D4) * (0.02 / (2 * L) ** 0.5))
            self.tr, self.sc = {}, {}
            self.sr, self.exact, self.cap = arm != "nvfp4_rtn", False, None

        def amax(self, key, t, head):
            st = shared["step"]
            if key not in self.tr or st < 24 or st % 8 == 0:
                mn, mx = torch.aminmax(t)
                self.tr[key] = max(abs(mn.item()), abs(mx.item()), 1e-30) if key != "y1" else max(mx.item(), 0.17)
            return self.tr[key] * head

        def S(self, key, v):
            if key not in self.sc or self.sc[key][0] != v:
                self.sc[key] = (v, torch.tensor([v], dtype=torch.float32, device="cuda"))
            return self.sc[key][1]

        def forward(self, x):
            if self.kind == "bf16":
                return F.linear(F.gelu(F.linear(x, self.w1)), self.w2)
            return Fn.apply(x, self.w1, self.w2, self)

    class Block(nn.Module):
        def __init__(self, i):
            super().__init__()
            kind = "nvfp4" if arm.startswith("nvfp4") else arm
            if kind == "nvfp4" and i == L - 1 and L >= 6:
                kind = "fp8"                                  # last sixth of the layers in 8 bits
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
            self.mlp = MLP(kind)

        def forward(self, x):
            B_, T_, _ = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
            q, k, v = [t.view(B_, T_, H, d // H).transpose(1, 2) for t in (q, k, v)]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T_, d)
            x = x + self.proj(a)
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte, self.wpe = nn.Embedding(V, d), nn.Embedding(T, d)
            self.blocks = nn.ModuleList([Block(i) for i in range(L)])
            self.lnf = nn.LayerNorm(d)
            self.head = nn.Linear(d, V, bias=False)
            self.head.weight = self.wte.weight
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
            for b in self.blocks:
                nn.init.normal_(b.proj.weight, mean=0.0, std=0.02 / (2 * L) ** 0.5)

        def forward(self, idx, targets=None):
            x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
            x = self.lnf(x)
            if targets is None:
                return self.head(x[:, -1:, :])
            return F.cross_entropy(self.head(x).float().view(-1, V), targets.reshape(-1))

    return GPT()


def get_batch(data, B, T, gen):
    import torch
    ix = torch.randint(0, data.numel() - T - 1, (B,), generator=gen, device=data.device)
    chunk = data[ix[:, None] + torch.arange(T + 1, device=data.device)[None, :]].long()
    return chunk[:, :-1], chunk[:, 1:]


@app.function(image=image, gpu="B200", cpu=8.0, memory=131072, timeout=3600, volumes={"/vol": vol})
def bench(widths: str = "2048,4096,8192", batch: int = 128, steps: int = 24, layers: int = 4, tag: str = "bench"):
    """Wall-clock per training step on one B200: PyTorch BF16, real FP8, real NVFP4 forward + FP8 backward."""
    import torch
    fb = FB()
    cfg = dict(CFG, n_layer=layers)
    vol.reload()
    data, _, cover = load_compact("train", "cuda", cfg["vocab"])
    out = dict(gpu=torch.cuda.get_device_name(0), batch=batch, tokens_per_step=batch * cfg["block"], layers=layers, vocab_coverage=cover, rows=[])
    sync = torch.cuda.synchronize
    for d in [int(v) for v in widths.split(",")]:
        row = dict(d=d)
        for arm in ["bf16", "fp8", "nvfp4"]:
            shared = dict(step=0)
            torch.manual_seed(1)
            model = build_real_model(arm, d, fb, shared, cfg).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
            gen = torch.Generator(device="cuda").manual_seed(5)
            ts, fwd, losses = [], [], []
            for s in range(steps + 10):
                shared["step"] = 100 + s          # past the every-step scale refresh; scales refresh every 8th step
                if s < 2:
                    shared["step"] = s
                x, y = get_batch(data, batch, cfg["block"], gen)
                sync(); t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                sync(); t1 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sync(); t2 = time.perf_counter()
                opt.step()
                sync(); t3 = time.perf_counter()
                if s >= 10:
                    ts.append((t3 - t0, t1 - t0, t2 - t1, t3 - t2))
                losses.append(loss.item())
            med = lambda i: round(1000 * sorted(t[i] for t in ts)[len(ts) // 2], 2)
            row[arm] = dict(step_ms=med(0), forward_ms=med(1), backward_ms=med(2), optimizer_ms=med(3), first_loss=round(losses[0], 3), last_loss=round(losses[-1], 3),
                            params=sum(p.numel() for p in model.parameters()))
            print(d, arm, row[arm], flush=True)
            del model, opt
            torch.cuda.empty_cache()
        for arm in ["fp8", "nvfp4"]:
            row[arm]["speedup_vs_bf16"] = round(row["bf16"]["step_ms"] / row[arm]["step_ms"], 3)
        row["nvfp4"]["speedup_vs_fp8"] = round(row["fp8"]["step_ms"] / row["nvfp4"]["step_ms"], 3)
        out["rows"].append(row)
        json.dump(out, open(f"/vol/real/{tag}.json", "w"), indent=1)
        vol.commit()
    return out


@app.function(image=image, gpu="B200", cpu=4.0, memory=32768, timeout=900, volumes={"/vol": vol})
def diag():
    """Where do the software and hardware NVFP4 kernels disagree?"""
    import torch
    fb = FB()
    torch.manual_seed(0)
    x = (torch.randn(4096, 1024, device="cuda") * torch.exp(torch.randn(4096, 1, device="cuda"))).bfloat16()
    ox = x.float().abs().max().item() / 2688
    c = {}
    for hw in (0, 1):
        fb.L.fb_set_hw(hw)
        c[hw] = fb.nvfp4(x, ox)["codes"]
    sh = torch.arange(16, device="cuda") * 4
    n0, n1 = (c[0][:, None] >> sh) & 15, (c[1][:, None] >> sh) & 15
    diff = n0 != n1
    import q4
    code, scale = q4.nvfp4(x.float(), parts=True)
    t = (x.float() / scale).reshape(-1, 16)
    idx = diff.nonzero()[:24]
    ex = [dict(t=t[i, j].item(), sw=int(n0[i, j]), hw=int(n1[i, j])) for i, j in idx.tolist()]
    tt = t[diff].abs()
    return dict(values_differing=diff.float().mean().item(), blocks_differing=diff.any(1).float().mean().item(), examples=ex,
                abs_t_min=tt.min().item() if tt.numel() else None, abs_t_max=tt.max().item() if tt.numel() else None,
                frac_abs_t_above_6=(tt > 6).float().mean().item() if tt.numel() else None)


@app.function(image=image, gpu="B200", cpu=8.0, memory=131072, timeout=5400, volumes={"/vol": vol})
def train_real(arm: str = "nvfp4", d: int = 4096, layers: int = 4, batch: int = 128, steps: int = 1500, lr: float = 2e-4,
               seed: int = 1, tag: str = "", save: bool = False):
    """Train TinyStories on one B200 with real GEMMs and log validation loss against wall-clock training time."""
    import math
    import numpy as np
    import tiktoken
    import torch
    fb = FB()
    fb.L.fb_set_copy_sr(0 if arm == "nvfp4_rtn" else 1)
    cfg = dict(CFG, n_layer=layers)
    V, T = cfg["vocab"], cfg["block"]
    tag = tag or f"{arm}_d{d}_s{seed}"
    vol.reload()
    data, top, cover = load_compact("train", "cuda", V)
    val, _, _ = load_compact("val", "cuda", V)
    lut = np.full(65536, V - 1, dtype=np.int64)
    lut[top] = np.arange(V - 1)
    shared = dict(step=0)
    torch.manual_seed(seed)
    model = build_real_model(arm, d, fb, shared, cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    n_mlp = sum(p.numel() for n_, p in model.named_parameters() if ".mlp." in n_)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=nodecay, weight_decay=0.0)], lr=lr, betas=(0.9, 0.95), fused=True)
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    warm = 100
    lr_at = lambda s: lr * (s + 1) / warm if s < warm else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    sync = torch.cuda.synchronize

    def evaluate(n):
        g2 = torch.Generator(device="cuda").manual_seed(7)
        tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(n):
                x, y = get_batch(val, batch, T, g2)
                tot += model(x, y).item()
        return tot / n

    log, clock, run, step_ms = [], 0.0, 0.0, []
    for s in range(steps + 1):
        if s % 100 == 0 or s == steps:
            v = evaluate(20)
            log.append(dict(step=s, train_s=round(clock, 2), val=round(v, 4), train=round(run, 4)))
            print(tag, log[-1], flush=True)
        if s == steps:
            break
        shared["step"] = s
        x, y = get_batch(data, batch, T, gen)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(s)
        sync(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(); dt = time.perf_counter() - t0
        li = loss.item()
        if not math.isfinite(li):
            log.append(dict(step=s, diverged=True)); break
        if s >= 10:
            clock += dt; step_ms.append(1000 * dt)
        run = li if s == 0 else 0.98 * run + 0.02 * li
    final_val = evaluate(100)
    enc = tiktoken.get_encoding("gpt2")
    eot = int(lut[50256])
    samples = []
    shared["step"] = 0
    for i, prompt in enumerate(["Once upon a time", "One day, a little girl named Lily", "Tom had a big red ball."]):
        ids = [int(lut[t]) for t in enc.encode_ordinary(prompt)]
        # fill the rest of the context with real validation stories that end just before the prompt, so the model
        # sees what it saw in training and every forward pass has the same shape
        ends = (val[2000:6000] == eot).nonzero()
        e = 2000 + int(ends[i % len(ends)].item())
        ctx_ids = val[e - (T - len(ids)) + 1:e + 1].tolist()
        idx = torch.tensor([ctx_ids + ids], device="cuda")
        g3 = torch.Generator(device="cuda").manual_seed(11 + i)
        new = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(150):
                logits = model(idx[:, -T:])[:, -1, :].float() / 0.8
                logits[:, V - 1] = -float("inf")
                vv, _ = torch.topk(logits, 50)
                logits[logits < vv[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g3)
                if nxt.item() == eot:
                    break
                new.append(nxt.item())
                idx = torch.cat([idx, nxt], 1)
        samples.append(dict(prompt=prompt, text=prompt + enc.decode([int(top[t]) for t in new])))
    step_ms.sort()
    res = dict(arm=arm, d=d, layers=layers, batch=batch, tokens_per_step=batch * T, steps=steps, lr=lr, seed=seed, tag=tag, gpu=torch.cuda.get_device_name(0),
               n_params=n_params, n_mlp_params=n_mlp, vocab=V, vocab_coverage=cover, log=log, final_val=round(final_val, 4), samples=samples,
               median_step_ms=round(step_ms[len(step_ms) // 2], 2) if step_ms else None, train_seconds=round(clock, 1))
    os.makedirs("/vol/real/runs", exist_ok=True)
    json.dump(res, open(f"/vol/real/runs/{tag}.json", "w"), indent=1)
    if save:
        os.makedirs("/vol/real/ckpt", exist_ok=True)
        torch.save(model.state_dict(), f"/vol/real/ckpt/{tag}.pt")
    vol.commit()
    print(tag, "final", final_val, res["median_step_ms"], samples[0]["text"][:200], flush=True)
    return res


@app.function(image=image, gpu="B200", cpu=8.0, memory=131072, timeout=3600, volumes={"/vol": vol})
def profile(widths: str = "4096,8192", batch: int = 128, layers: int = 4, steps: int = 6, tag: str = "profile"):
    """Per-operation time inside a live training step. Every timed operation is bracketed by a GPU synchronize, so the
    parts add up to more than an untimed step; use it for shares, and bench() for totals."""
    import torch
    fb = FB()
    cfg = dict(CFG, n_layer=layers)
    vol.reload()
    data, _, _ = load_compact("train", "cuda", cfg["vocab"])
    sync = torch.cuda.synchronize
    acc = {}
    real_gemm, real_nv, real_f8 = fb.gemm, fb.nvfp4, fb.fp8
    state = dict(on=False, phase="fwd")

    def timed(name, fn):
        def w(*a, **k):
            if not state["on"]:
                return fn(*a, **k)
            sync(); t0 = time.perf_counter(); r = fn(*a, **k); sync()
            key = f"{state['phase']}_{name(*a, **k)}"
            acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0
            return r
        return w

    fb.gemm = timed(lambda kind, *a, **k: ["bf16", "fp8", "nvfp4"][kind] + "_gemm", real_gemm)
    fb.nvfp4 = timed(lambda x, outer, f8_scale=None, gelu=False: ("gelu_" if gelu else "") + ("w_" if x.dtype == torch.float32 else "") + "quant_nvfp4", real_nv)
    fb.fp8 = timed(lambda x, scale, sr=True, gelu=False, grad_of=None: ("gelu_" if gelu else "") + ("gelubwd_" if grad_of is not None else "") + ("w_" if x.dtype == torch.float32 else "") + "quant_fp8", real_f8)
    out = dict(gpu=torch.cuda.get_device_name(0), batch=batch, layers=layers, rows=[])
    for d in [int(v) for v in widths.split(",")]:
        row = dict(d=d)
        for arm in ["bf16", "fp8", "nvfp4"]:
            shared = dict(step=0)
            torch.manual_seed(1)
            model = build_real_model(arm, d, fb, shared, cfg).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
            gen = torch.Generator(device="cuda").manual_seed(5)
            mt = dict(mlp_fwd=0.0, mlp_bwd=0.0)
            tb = {}

            def pre(m, a):
                if state["on"]:
                    sync(); tb[id(m)] = time.perf_counter()

            def post(m, a, o):
                if state["on"]:
                    sync(); mt["mlp_fwd"] += time.perf_counter() - tb[id(m)]

            def bpre(m, go):
                if state["on"]:
                    sync(); tb[("b", id(m))] = time.perf_counter()

            def bpost(m, gi, go):
                if state["on"]:
                    sync(); mt["mlp_bwd"] += time.perf_counter() - tb[("b", id(m))]

            for b in model.blocks:
                b.mlp.register_forward_pre_hook(pre); b.mlp.register_forward_hook(post)
                b.mlp.register_full_backward_pre_hook(bpre); b.mlp.register_full_backward_hook(bpost)
            acc.clear()
            tot = dict(fwd=0.0, bwd=0.0, opt=0.0)
            for s in range(steps + 6):
                shared["step"] = 101 + s if s >= 2 else s        # never a scale-refresh step while profiling
                state["on"] = s >= 6
                x, y = get_batch(data, batch, cfg["block"], gen)
                sync(); t0 = time.perf_counter()
                state["phase"] = "fwd"
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                sync(); t1 = time.perf_counter()
                state["phase"] = "bwd"
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sync(); t2 = time.perf_counter()
                opt.step()
                sync(); t3 = time.perf_counter()
                if state["on"]:
                    tot["fwd"] += t1 - t0; tot["bwd"] += t2 - t1; tot["opt"] += t3 - t2
            state["on"] = False
            ms = lambda v: round(1000 * v / steps, 3)
            row[arm] = dict(forward_ms=ms(tot["fwd"]), backward_ms=ms(tot["bwd"]), optimizer_ms=ms(tot["opt"]), mlp_forward_ms=ms(mt["mlp_fwd"]),
                            mlp_backward_ms=ms(mt["mlp_bwd"]), ops={k: ms(v) for k, v in sorted(acc.items())})
            print(d, arm, json.dumps(row[arm]), flush=True)
            del model, opt
            torch.cuda.empty_cache()
        out["rows"].append(row)
        json.dump(out, open(f"/vol/real/{tag}.json", "w"), indent=1)
        vol.commit()
    return out


@app.function(image=image, gpu="B200", cpu=8.0, memory=65536, timeout=600)
def specs():
    """What the machine we rent actually is."""
    import torch
    q = "name,driver_version,memory.total,power.max_limit,power.default_limit,clocks.max.sm,clocks.max.memory,pci.link.gen.max,pci.link.width.max,compute_cap"
    smi = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"], capture_output=True, text=True).stdout
    p = torch.cuda.get_device_properties(0)
    cpu = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Socket|Core' ; free -g | head -2; nvidia-smi | head -12; nvcc --version | tail -2", shell=True, capture_output=True, text=True).stdout
    out = dict(smi=smi, torch_name=p.name, sms=p.multi_processor_count, total_memory_bytes=p.total_memory, capability=f"{p.major}.{p.minor}",
               torch=torch.__version__, cuda_in_torch=torch.version.cuda, host=cpu, n_gpus=torch.cuda.device_count(), cpus_visible=os.cpu_count())
    os.makedirs("/tmp/x", exist_ok=True)
    return out


@app.function(image=image, gpu="B200", cpu=8.0, memory=131072, timeout=3600, volumes={"/vol": vol})
def analyze_real(d: int = 4096, layers: int = 4, batch: int = 128, recipe_tag: str = "nvfp4_d4096_s1", base_tag: str = "bf16_d4096_s1", layer: int = 1,
                 only_c: bool = False):
    """The measurements behind the explanatory figures, taken on the B200-trained wide model."""
    import torch
    import torch.nn.functional as F
    import q4
    fb = FB()
    cfg = dict(CFG, n_layer=layers)
    V, T = cfg["vocab"], cfg["block"]
    vol.reload()
    data, _, _ = load_compact("train", "cuda", V)
    val, _, _ = load_compact("val", "cuda", V)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    zero_frac = lambda q, x: ((q == 0) & (x != 0)).float().mean().item()
    out = dict(gpu=torch.cuda.get_device_name(0), d=d, layers=layers, tokens=batch * T)

    def load(arm, tag):
        sh = dict(step=0)
        m = build_real_model(arm, d, fb, sh, cfg).cuda()
        m.load_state_dict(torch.load(f"/vol/real/ckpt/{tag}.pt", map_location="cuda"))
        return m, sh

    def val_loss(m, n=40):
        g = torch.Generator(device="cuda").manual_seed(7)
        tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(n):
                x, y = get_batch(val, batch, T, g)
                tot += m(x, y).item()
        return tot / n

    def bias_experiment(model, shared, mlps, xb, yb):
        """Weight gradients of the LAST block's MLP: its incoming gradient is exactly repeatable (PyTorch's attention
        backward is not, and that would blur the comparison for earlier blocks). Real 4-bit forward, real 8-bit backward."""
        last = mlps[-1]

        def g_last():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_ = model(xb, yb)
            gs = torch.autograd.grad(l_, [last.w1, last.w2])
            return torch.cat([t.float().flatten() for t in gs])

        fb.L.fb_set_hw(0)                   # software rounding, so the codes equal q4's and the reference is exact
        shared["step"] = 0
        last.exact = True
        ref = g_last()
        cos = lambda a, b: (a @ b / (a.norm() * b.norm())).item()
        last.exact, last.sr = False, False
        fb.L.fb_set_copy_sr(0)
        g_last()                            # settle the gradient scales and the GEMM plans
        shared["step"] = 101
        a = g_last()
        C = dict(fp8_rtn=dict(mlp_rel_err=rel(a, ref), mlp_cos=cos(a, ref), repeat_rel_diff=rel(g_last(), a)))
        last.sr = True
        fb.L.fb_set_copy_sr(1)
        acc, curve = torch.zeros_like(ref), {}
        for i_ in range(1, 65):
            acc += g_last()
            if i_ in (1, 2, 4, 8, 16, 32, 64):
                curve[i_] = dict(mlp_rel_err=rel(acc / i_, ref), mlp_cos=cos(acc / i_, ref))
        C["fp8_sr_mean_of_n"] = curve
        fb.L.fb_set_hw(1)
        print("C", C["fp8_rtn"], {k: round(v["mlp_rel_err"], 5) for k, v in curve.items()}, flush=True)
        return C

    if only_c:
        out = json.load(open("/vol/real/analysis.json"))
        model, shared = load("nvfp4", recipe_tag)
        mlps = [b.mlp for b in model.blocks]
        gen = torch.Generator(device="cuda").manual_seed(123)
        xb, yb = get_batch(data, batch, T, gen)
        out["C_grad_bias"] = bias_experiment(model, shared, mlps, xb, yb)
        json.dump(out, open("/vol/real/analysis.json", "w"))
        vol.commit()
        return out["C_grad_bias"]

    F_ = {}
    for name, arm, tag in [("bf16 model, bf16 forward", "bf16", base_tag), ("bf16 model, recipe forward (quantized after training)", "nvfp4", base_tag),
                           ("recipe model, bf16 forward (quantization removed)", "bf16", recipe_tag), ("recipe model, recipe forward", "nvfp4", recipe_tag)]:
        m, _ = load(arm, tag)
        F_[name] = round(val_loss(m), 4)
        if name != "recipe model, recipe forward":
            del m
            torch.cuda.empty_cache()
    out["F_ptq"] = F_
    print("F", F_, flush=True)
    model, shared = m, _
    mlps = [b.mlp for b in model.blocks]
    mlp = mlps[layer]
    gen = torch.Generator(device="cuda").manual_seed(123)
    xb, yb = get_batch(data, batch, T, gen)

    def grads():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_ = model(xb, yb)
        gs = torch.autograd.grad(l_, [p for q_ in mlps for p in (q_.w1, q_.w2)])
        return torch.cat([t.float().flatten() for t in gs])

    mlp.cap = {}
    grads()
    cap, mlp.cap = mlp.cap, None
    tensors = dict(weight_fc=mlp.w1.detach().float(), weight_out=mlp.w2.detach().float(), act_into_fc=cap["x"].float(),
                   act_into_out=F.gelu(cap["y1"].float()), grad_at_mlp_output=cap["g"].float())

    def fp4_one_scale(x):
        s = x.abs().max() / 6.0
        return q4.round_float(x / s, q4.E2M1) * s

    A = {}
    for name, x in tensors.items():
        x = x.contiguous()
        row = dict(shape=list(x.shape), absmax=x.abs().max().item(), rms=x.pow(2).mean().sqrt().item())
        for k, fn in [("fp4_one_scale", lambda: fp4_one_scale(x)), ("nvfp4_1x16", lambda: q4.nvfp4(x)), ("nvfp4_1x16_per_row_outer", lambda: q4.nvfp4(x, outer="row")),
                      ("nvfp4_1x16_per_row_outer_4over6", lambda: q4.nvfp4(x, outer="row", four_six=True)),
                      ("nvfp4_16x1_other_dim", lambda: q4.nvfp4(x.t().contiguous()).t()), ("nvfp4_16x16_tile", lambda: q4.nvfp4(x, tile=True)),
                      ("mxfp8", lambda: q4.mxfp8(x)), ("bf16", lambda: x.bfloat16().float())]:
            qv = fn()
            row[k] = dict(rel_err=rel(qv, x), to_zero=zero_frac(qv, x))
            del qv
        A[name] = row
        print("A", name, {k: (round(v["rel_err"], 4), round(v["to_zero"], 3)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    out["A_formats"] = A

    Bm = {}
    for wname in ["weight_fc"]:
        w = tensors[wname]
        wq = q4.nvfp4(w)
        wqt = wq.t().contiguous()
        r = dict(forward_err_vs_fp32=rel(wq, w))
        for fmt, fn in [("nvfp4", lambda m_: q4.nvfp4(wqt, m_)), ("mxfp8", lambda m_: q4.mxfp8(wqt, m_))]:
            rt = fn("rtn")
            r[f"{fmt}_rtn"] = dict(mismatch=rel(rt, wqt), to_zero=zero_frac(rt, wqt), abs_shrink=(rt.abs().mean() / wqt.abs().mean()).item())
            acc, curve = torch.zeros_like(wqt), {}
            for i in range(1, 257):
                acc += fn("sr")
                if i in (1, 4, 16, 64, 256):
                    curve[i] = rel(acc / i, wqt)
            r[f"{fmt}_sr_mean_of_n"] = curve
        Bm[wname] = r
    out["B_transpose"] = Bm
    print("B", json.dumps(Bm)[:500], flush=True)

    out["C_grad_bias"] = bias_experiment(model, shared, mlps, xb, yb)

    y = cap["y"].float().contiguous()
    yr = q4.rht(y)
    blocks, blocks_r = y.reshape(-1, 32), yr.reshape(-1, 32)
    crest = lambda b: (b.abs().amax(1) / b.pow(2).mean(1).sqrt().clamp_min(1e-20))
    kurt = lambda t: ((t - t.mean()).pow(4).mean() / t.var().pow(2)).item()
    D = dict(shape=list(y.shape), kurtosis_before=kurt(y), kurtosis_after_rht=kurt(yr), max_over_rms_per_block_before=crest(blocks).mean().item(),
             max_over_rms_per_block_after=crest(blocks_r).mean().item(),
             err=dict(mxfp8=rel(q4.mxfp8(y), y), s2f6_no_rht=rel(q4.s2f6(y), y), rht_s2f6=rel(q4.combine_quant(y, "rht_s2f6"), y), bf16=rel(y.bfloat16().float(), y)))
    i = int(torch.argmax(crest(blocks)).item())
    D["example_block_worst"] = dict(before=blocks[i].tolist(), after=blocks_r[i].tolist())
    out["D_wire"] = D
    print("D", {k: v for k, v in D.items() if not k.startswith("example")}, flush=True)

    w = tensors["weight_fc"]
    N, K = w.shape
    tiles = w.reshape(N // 16, 16, K // 16, 16).permute(0, 2, 1, 3).reshape(-1, 16, 16)
    row_amax = tiles.abs().amax(2)
    spread = row_amax.amax(1) / row_amax.median(1).values
    E = {}
    s_out = w.abs().max() / (6.0 * 448.0)
    for label, idx in [("most_uneven", int(torch.argmax(spread).item()))]:
        t = tiles[idx].contiguous()

        def qt(tt, mode):
            am = tt.abs().amax(1, keepdim=True) if mode == "rows" else (tt.abs().amax(0, keepdim=True) if mode == "cols" else tt.abs().max().reshape(1, 1))
            sb = q4.round_float(am / 6.0 / s_out, q4.E4M3).clamp_min(2.0 ** -9) * s_out
            return q4.round_float(tt / sb, q4.E2M1) * sb

        E[label] = dict(tile_index=idx, values=t.tolist(), nonzero=int((t != 0).sum().item()),
                        **{k: dict(q=qt(t, k).tolist(), survive=int((qt(t, k) != 0).sum().item())) for k in ("rows", "cols", "tile")})
    out["E_tile"] = E
    code, scale = q4.nvfp4(w, parts=True)
    out["worked_block"] = dict(values=w[5, :16].tolist(), codes=code[5, :16].tolist(), scale=scale[5, 0].item())
    out["weight_hist"] = dict(std=w.std().item(), absmax=w.abs().max().item())
    json.dump(out, open("/vol/real/analysis.json", "w"))
    vol.commit()
    return {k: out[k] for k in ("F_ptq", "B_transpose", "C_grad_bias")}


@app.function(image=image, gpu="B200", cpu=8.0, memory=131072, timeout=1800, volumes={"/vol": vol})
def detcheck(d: int = 4096, layers: int = 4, batch: int = 128, tag: str = "nvfp4_d4096_s1"):
    """Is a backward pass with every operand rounded to nearest repeatable? Finds which piece is not."""
    import torch
    fb = FB()
    cfg = dict(CFG, n_layer=layers)
    vol.reload()
    data, _, _ = load_compact("train", "cuda", cfg["vocab"])
    shared = dict(step=0)
    model = build_real_model("nvfp4", d, fb, shared, cfg).cuda()
    model.load_state_dict(torch.load(f"/vol/real/ckpt/{tag}.pt", map_location="cuda"))
    mlps = [b.mlp for b in model.blocks]
    gen = torch.Generator(device="cuda").manual_seed(123)
    xb, yb = get_batch(data, batch, cfg["block"], gen)
    rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

    def run():
        for q_ in mlps:
            q_.cap = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_ = model(xb, yb)
        gs = torch.autograd.grad(l_, [p for q_ in mlps for p in (q_.w1, q_.w2)])
        caps = [dict(q_.cap) for q_ in mlps]
        for q_ in mlps:
            q_.cap = None
        return l_.item(), [t.float() for t in gs], caps

    out = {}
    for hw in (0, 1):
        fb.L.fb_set_hw(hw)
        for name, exact, sr, csr in [("exact", True, True, 1), ("nearest", False, False, 0), ("stochastic", False, True, 1)]:
            for q_ in mlps:
                q_.exact, q_.sr = exact, sr
            fb.L.fb_set_copy_sr(csr)
            shared["step"] = 0
            run()                                  # settle scales and plans
            shared["step"] = 101
            l1, g1, c1 = run()
            l2, g2, c2 = run()
            out[f"hw{hw}_{name}"] = dict(loss_same=l1 == l2, grad_w1_per_layer=[rel(g2[2 * i], g1[2 * i]) for i in range(layers)],
                                         grad_w2_per_layer=[rel(g2[2 * i + 1], g1[2 * i + 1]) for i in range(layers)],
                                         incoming_g_per_layer=[rel(c2[i]["g"].float(), c1[i]["g"].float()) for i in range(layers)],
                                         y_per_layer=[rel(c2[i]["y"].float(), c1[i]["y"].float()) for i in range(layers)])
            print(f"hw{hw}_{name}", json.dumps(out[f"hw{hw}_{name}"]), flush=True)
    return out
