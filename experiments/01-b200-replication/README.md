# 01: Replicating 4-bit training on one NVIDIA B200

**Question.** Vizuara AI Labs published a test of 4-bit training on a single B200
([code](https://github.com/VizuaraAI/4bit-training-b200), [article](https://4bit.vizuara.ai)). Do we get the same
numbers when we run their code ourselves?

**Status.** In progress. The arithmetic checks and single-multiply timings are done and reproduce. The full-step
benchmark and the training runs are submitted and waiting for B200 capacity; this page has no training results yet.

All of our runs: one B200 rented from Modal, 18 November 2026. The self-test ran on an Apple M1 Pro CPU.

## Results so far

Every number below is our own measurement, from the JSON files in `results/`. The last column says how it compares
with what upstream published.

| Check | Our result | Against upstream |
|:--|--:|:--|
| Format self-test (`q4.py`): value grids, worked example | pass | same |
| 8-bit multiply against reference: DGRAD / WGRAD / FPROP | 1.27e-7 / 2.74e-7 / 5.73e-8 | same |
| 4-bit multiply against unquantized | 0.13452 | same |
| 4-bit multiply against reference, software quantize kernels | 6.08e-8 | same |
| 4-bit multiply against reference, hardware quantize kernels | 0.00627 | same |
| One 4-bit multiply vs BF16, width 2,048 | 3.13x faster | about the same |
| One 4-bit multiply vs BF16, width 4,096 | 3.01x faster | ours is lower |
| One 4-bit multiply vs BF16, width 8,192 | 3.28x faster | ours is lower |
| Hardware quantize kernels vs software ones, nine kernels | 1.43x to 4.93x faster | same |

Data set: 658,557 training stories (144.7 million tokens) and 21,990 validation stories.

## What we found

**The headline accuracy figure describes kernels that training does not use.** The upstream README says the real
4-bit multiply matches the reference arithmetic "to 6 parts in 100 million". We reproduce that figure exactly, but
only with the software quantize kernels. The code as published defaults to the kernels built on Blackwell's
conversion instructions, and those are what the training runs use. With them the multiply is 0.63% away from the
reference, because the two kernels pick different codes for 1.1% of activation blocks. Weight codes and block scales
are identical between the two. Whether that difference moves the training loss is not yet tested.

**Single-multiply speedups are a little lower for us at larger widths.** We measure 3.0x and 3.3x at widths 4,096
and 8,192, where upstream reports about 3.4x and 3.6x. These are single timing runs on a different rented machine, so
this may be machine-to-machine variation.

**The kernel speed claim holds for eight of nine kernels.** Upstream says the hardware kernels are "2 to 5x faster".
We measure 2.1x to 4.9x for eight of them. The ninth, the GELU-backward kernel, is 1.4x.

## Still to run

Submitted, waiting for a B200:

* `bench`: one full training step for BF16, all-8-bit, and 4-bit forward with 8-bit backward, at three widths.
* Four training runs at width 4,096, 2,000 steps, seed 1: `bf16`, `fp8`, `nvfp4`, `nvfp4_rtn`.

Not yet submitted: `profile`, second seeds, the three width-8,192 runs, `analyze_real`, `detcheck`.

## Files

| Path | What it is | Whose |
|:--|:--|:--|
| `results/` | Everything we measured: `selftest.json`, `prep.json`, `probe.json`, `kprobe.json`. `calls.txt` lists the Modal call IDs of the submitted runs. | Ours |
| `fb.cu`, `real.py`, `q4.py`, `train.py`, `mo.py` | The CUDA library, the model and Modal functions, the reference arithmetic, data prep, and the launcher. | Vizuara, unmodified |

Upstream's published result files are not part of this repository. To compare against them, clone the upstream
repository; a local `reference/` folder here is git-ignored for that purpose.

## Running it

You need a [Modal](https://modal.com) account that can rent a B200. From this folder:

```bash
pip install modal
python3 mo.py modal deploy train.py
python3 mo.py call prep                       # download and tokenize TinyStories once
python3 mo.py modal deploy real.py
export FOURBIT_APP=fourbit-real
python3 mo.py call probe                      # hardware against arithmetic, first timings
python3 mo.py call kprobe                     # software kernels against hardware kernels
python3 mo.py call bench                      # full training steps at three widths
python3 mo.py spawn train_real arm=nvfp4 d=4096 steps=2000 lr=0.0002
```

`python3 q4.py` runs the format self-test on any machine with PyTorch.

## How this code differs from the Thinking Machines recipe

This matters when reading any result here as a test of the blog's recipe:

* The backward pass uses 8-bit floats with one scale per tensor, not MXFP8 with a scale per 32 values.
* Activations get one outer scale per tensor, not one per token.
* No Four Over Six.
* Only the MLP layers of a four-block dense GPT are quantized. The recipe targets mixture-of-experts layers.
* One GPU, so nothing is compressed between GPUs.
* The multiplies are NVIDIA's cuBLASLt, not custom kernels.

## Credits

The code in this folder was written by Dr. Raj Dandekar and the team at [Vizuara AI Labs](https://vizuara.ai) and is
taken unmodified from [VizuaraAI/4bit-training-b200](https://github.com/VizuaraAI/4bit-training-b200) at commit
`135585e`. The measurements in `results/` and the findings above are ours.
