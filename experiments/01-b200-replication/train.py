"""train.py: a nanoGPT-style model on TinyStories, trained in BF16 and then with the quantized layer from q4.py.

    python3 mo.py modal deploy train.py
    python3 mo.py call selftest
    python3 mo.py call prep
    python3 mo.py spawn train arm=recipe seed=1

Every run writes /vol/runs/<arm>_s<seed>/result.json (loss log, final validation loss, sampled stories) and ckpt.pt.
The quantization is simulated in FP32 on whatever GPU Modal gives us: the values are exactly the ones NVFP4 and MXFP8
can store, so the loss curves are the real experiment, and the step time is not.
"""
import json
import math
import os
import time

import modal

app = modal.App("fourbit-tinystories")
vol = modal.Volume.from_name("fourbit-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "numpy==2.3.5", "tiktoken==0.9.0")
         .env({"PYTHONUNBUFFERED": "1"})
         .add_local_python_source("q4"))

HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
EOT = 50256
MODEL = dict(n_layer=6, n_head=6, n_embd=384, block=256, vocab=50257)
TRAIN = dict(steps=4000, batch=64, lr=1e-3, lr_min=1e-4, warmup=200, wd=0.1, clip=1.0, eval_every=250, eval_batches=40)
PROMPTS = ["Once upon a time", "One day, a little girl named Lily", "Tom had a big red ball."]


def arms():
    """The experiment arms. Each maps a block index to (config for the two MLP layers, config for attention layers)."""
    from q4 import cfg
    none = cfg()

    def mk(fx="nvfp4", fw="nvfp4", four_six=True, bfmt="mxfp8", bround="sr", keep_w=False, combine="rht_s2f6",
           last_mx=True, attn=False):
        main = cfg(fx, fw, four_six, bfmt, bround, keep_w, combine)
        last = cfg("mxfp8", "mxfp8", False, bfmt, bround, False, combine) if last_mx else main
        nocomb = lambda c: cfg(c.fx, c.fw, c.four_six, c.bfmt, c.bround, c.keep_w, "none")
        return lambda i, n: ((last if i == n - 1 else main), (nocomb(last if i == n - 1 else main) if attn else none))

    return {
        "bf16": lambda i, n: (none, none),
        # the article's recipe: NVFP4 forward (4/6 on activations), MXFP8 backward with stochastic re-quantization,
        # RHT + S2F6 on the layer's output, the last sixth of the layers (1 of 6) in MXFP8
        "recipe": mk(),
        "recipe_rtn": mk(bround="rtn"),                       # the same, but re-quantize to nearest
        "fwd_only": mk(bfmt="exact"),                         # NVFP4 forward, high-precision backward
        "fp4bwd_rtn": mk(bfmt="nvfp4", bround="rtn"),         # NVFP4 backward too, round to nearest: "full NVFP4"
        "fp4bwd_sr": mk(bfmt="nvfp4", bround="sr"),           # NVFP4 backward with stochastic rounding
        "tile16": mk(fw="nvfp4_tile", keep_w=True),           # NVIDIA's 16 x 16 tiles for weights, no re-quantization
        "mxfp8": mk(fx="mxfp8", fw="mxfp8", four_six=False, combine="none"),   # everything in MXFP8
        "recipe_mx8comb": mk(combine="mxfp8"),                # recipe, but the output crosses the wire in MXFP8
        "recipe_all": mk(attn=True),                          # recipe on the attention projections as well
    }


# ------------------------------------------------------------------------------------------------------------- data
@app.function(image=image, cpu=16.0, memory=32768, timeout=2400, volumes={"/vol": vol})
def prep(max_bytes: int = 600_000_000):
    """Download TinyStories, tokenize with the GPT-2 tokenizer, and store uint16 token files on the volume."""
    import urllib.request
    import numpy as np
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs("/vol/data", exist_ok=True)
    out = {}
    for split, fname, cap in [("val", "TinyStories-valid.txt", 10 ** 10), ("train", "TinyStories-train.txt", max_bytes)]:
        t0 = time.time()
        buf = bytearray()
        with urllib.request.urlopen(urllib.request.Request(HF + fname, headers={"User-Agent": "python-urllib"})) as r:
            while len(buf) < cap:
                chunk = r.read(1 << 24)
                if not chunk:
                    break
                buf += chunk
        text = bytes(buf[:cap]).decode("utf-8", errors="ignore")
        stories = [s.strip() for s in text.split("<|endoftext|>")]
        stories = [s for s in stories if s][:-1] if len(buf) >= cap else [s for s in stories if s]
        parts = []
        for i in range(0, len(stories), 20000):
            for ids in enc.encode_ordinary_batch(stories[i:i + 20000], num_threads=16):
                parts.append(np.asarray(ids + [EOT], dtype=np.uint16))
        arr = np.concatenate(parts)
        arr.tofile(f"/vol/data/{split}.bin")
        out[split] = dict(bytes=len(buf), stories=len(stories), tokens=int(arr.size), seconds=round(time.time() - t0, 1),
                          distinct_tokens=int(np.unique(arr).size))
        print(split, out[split], flush=True)
    json.dump(out, open("/vol/data/meta.json", "w"), indent=1)
    vol.commit()
    return out


@app.function(image=image, cpu=4.0, memory=8192, timeout=900)
def selftest():
    import q4
    return q4.selftest()


# ------------------------------------------------------------------------------------------------------------ model
def build_model(arm):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from q4 import QLinear, combine_quant
    pick = arms()[arm]
    L, H, D, T, V = MODEL["n_layer"], MODEL["n_head"], MODEL["n_embd"], MODEL["block"], MODEL["vocab"]

    class Block(nn.Module):
        def __init__(self, i):
            super().__init__()
            cm, ca = pick(i, L)
            self.cm = cm
            self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.qkv, self.proj = QLinear(D, 3 * D, ca), QLinear(D, D, ca)
            self.fc, self.out = QLinear(D, 4 * D, cm), QLinear(4 * D, D, cm)      # the "expert": the layers we quantize

        def forward(self, x):
            B, T_, _ = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
            q, k, v = [t.view(B, T_, H, D // H).transpose(1, 2) for t in (q, k, v)]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T_, D)
            x = x + self.proj(a)
            y = self.out(F.gelu(self.fc(self.ln2(x))))
            if self.cm.combine != "none":                    # the output crosses the wire quantized (straight-through)
                with torch.autocast("cuda", enabled=False):
                    yq = combine_quant(y.float(), self.cm.combine)
                y = y + (yq.to(y.dtype) - y).detach()
            return x + y

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte, self.wpe = nn.Embedding(V, D), nn.Embedding(T, D)
            self.blocks = nn.ModuleList([Block(i) for i in range(L)])
            self.lnf = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)
            self.head.weight = self.wte.weight
            self.apply(self._init)
            for n_, p in self.named_parameters():
                if n_.endswith("proj.weight") or n_.endswith("out.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * L))

        def _init(self, m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
            x = self.lnf(x)
            if targets is None:
                return self.head(x[:, -1:, :])
            logits = self.head(x)
            return F.cross_entropy(logits.float().view(-1, V), targets.reshape(-1))

    return GPT()


def load_tokens(split, device):
    import numpy as np
    import torch
    return torch.from_numpy(np.fromfile(f"/vol/data/{split}.bin", dtype=np.uint16).astype(np.int32)).to(device)


def get_batch(data, B, T, gen):
    import torch
    ix = torch.randint(0, data.numel() - T - 1, (B,), generator=gen, device=data.device)
    pos = ix[:, None] + torch.arange(T + 1, device=data.device)[None, :]
    chunk = data[pos].long()
    return chunk[:, :-1], chunk[:, 1:]


def sample(model, enc, prompt, n_new=170, temperature=0.8, top_k=50, seed=0):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    idx = torch.tensor([enc.encode_ordinary(prompt)], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n_new):
            logits = model(idx[:, -MODEL["block"]:])[:, -1, :].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
            if nxt.item() == EOT:
                break
            idx = torch.cat([idx, nxt], 1)
    return enc.decode(idx[0].tolist())


@app.function(image=image, gpu=["H100", "A100-80GB", "A100-40GB", "L40S"], cpu=4.0, memory=24576, timeout=5400,
              volumes={"/vol": vol})
def train(arm: str = "bf16", seed: int = 1, steps: int = 0, tag: str = ""):
    import torch
    import tiktoken
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfgt = dict(TRAIN)
    if steps:
        cfgt["steps"] = steps
    tag = tag or f"{arm}_s{seed}"
    out_dir = f"/vol/runs/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    train_data, val_data = load_tokens("train", dev), load_tokens("val", dev)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = build_model(arm).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_mlp = sum(p.numel() for n_, p in model.named_parameters() if ".fc." in n_ or ".out." in n_)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfgt["wd"]), dict(params=nodecay, weight_decay=0.0)],
                            lr=cfgt["lr"], betas=(0.9, 0.95))
    B, T = cfgt["batch"], MODEL["block"]
    gen = torch.Generator(device=dev).manual_seed(1000 + seed)

    def lr_at(s):
        if s < cfgt["warmup"]:
            return cfgt["lr"] * (s + 1) / cfgt["warmup"]
        p = (s - cfgt["warmup"]) / max(1, cfgt["steps"] - cfgt["warmup"])
        return cfgt["lr_min"] + 0.5 * (cfgt["lr"] - cfgt["lr_min"]) * (1 + math.cos(math.pi * p))

    def evaluate(n):
        model.eval()
        g2 = torch.Generator(device=dev).manual_seed(7)      # the same validation batches for every arm and every step
        tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(n):
                x, y = get_batch(val_data, B, T, g2)
                tot += model(x, y).item()
        model.train()
        return tot / n

    log, t0, run = [], time.time(), 0.0
    step_times = []
    for s in range(cfgt["steps"] + 1):
        if s % cfgt["eval_every"] == 0:
            v = evaluate(cfgt["eval_batches"])
            log.append(dict(step=s, val=round(v, 4), train=round(run, 4), t=round(time.time() - t0, 1)))
            print(tag, log[-1], gpu, flush=True)
            json.dump(dict(arm=arm, seed=seed, log=log, gpu=gpu, done=False), open(f"{out_dir}/result.json", "w"))
            vol.commit()
        if s == cfgt["steps"]:
            break
        ts = time.time()
        x, y = get_batch(train_data, B, T, gen)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfgt["clip"])
        opt.step()
        li = loss.item()
        if not math.isfinite(li):
            log.append(dict(step=s, val=float("nan"), train=float("nan"), t=round(time.time() - t0, 1), diverged=True))
            break
        run = li if s == 0 else 0.98 * run + 0.02 * li
        if s >= 20:
            step_times.append(time.time() - ts)
    final_val = evaluate(200)
    enc = tiktoken.get_encoding("gpt2")
    samples = [dict(prompt=p, text=sample(model, enc, p, seed=11 + i)) for i, p in enumerate(PROMPTS)]
    step_times.sort()
    res = dict(arm=arm, seed=seed, tag=tag, gpu=gpu, n_params=n_params, n_mlp_params=n_mlp, model=MODEL, train=cfgt,
               log=log, final_val=round(final_val, 4), samples=samples, done=True,
               median_step_ms=round(1000 * step_times[len(step_times) // 2], 2) if step_times else None,
               wall_s=round(time.time() - t0, 1))
    json.dump(res, open(f"{out_dir}/result.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{out_dir}/ckpt.pt")
    vol.commit()
    print(tag, "final", final_val, samples[0]["text"][:300], flush=True)
    return res


@app.function(image=image, cpu=1.0, timeout=300, volumes={"/vol": vol})
def status():
    vol.reload()
    out = {}
    if os.path.isdir("/vol/runs"):
        for d in sorted(os.listdir("/vol/runs")):
            p = f"/vol/runs/{d}/result.json"
            if os.path.exists(p):
                r = json.load(open(p))
                last = r["log"][-1] if r.get("log") else {}
                out[d] = dict(done=r.get("done"), step=last.get("step"), val=last.get("val"), train=last.get("train"),
                              t=last.get("t"), final=r.get("final_val"), gpu=r.get("gpu"), ms=r.get("median_step_ms"))
    return out


@app.function(image=image, cpu=1.0, timeout=300, volumes={"/vol": vol})
def fetch(path: str):
    vol.reload()
    return open(f"/vol/{path}").read()
