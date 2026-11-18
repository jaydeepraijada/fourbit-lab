# 01: Replicating 4-bit training on one NVIDIA B200

**Question.** Vizuara AI Labs published a test of 4-bit training on a single B200
([code](https://github.com/VizuaraAI/4bit-training-b200), [article](https://4bit.vizuara.ai)). Do we get the same
numbers when we run their code ourselves?

**Answer.** The speed reproduces. The loss results reproduce only in part: 4-bit training is consistently a little
worse than BF16, but the gap is about three times larger in our runs, and two of the published comparisons flip
direction when rerun.

**Status.** Complete. Every published run has been repeated.

All of our runs: one B200 rented from Modal, 18 November 2026. The self-test ran on an Apple M1 Pro CPU.

The model is a four-block GPT trained on TinyStories, with only the MLP layers quantized. "4-bit" below means 4-bit
forward multiplies and 8-bit backward multiplies; "8-bit" means 8-bit both ways. Loss is validation loss.

## What we found

**4-bit trains 1.3x to 1.4x faster than BF16, but only 2% to 8% faster than 8-bit.** At width 2,048 it was no faster
than 8-bit at all. Almost all of the gain over BF16 comes from leaving 16 bits, not from going down to 4.

**Quantizing costs more than the 4-bit multiply saves, at width 4,096.** The 4-bit forward multiplies take 6.3 ms
per step against 10.7 ms for 8-bit, but preparing their inputs takes 8.8 ms against 3.7 ms. The 4-bit MLP forward
pass ends up slower than the 8-bit one (18.1 ms against 17.6 ms). At width 8,192 the multiply saving wins, and the
MLP forward pass is 16% faster. The MLP is about 39% of a step, which caps the gain in any case.

**4-bit costs loss, and the cost is larger for us than upstream reports.** At width 4,096, averaged over two seeds,
4-bit ends 0.0096 above BF16 (1.6936 against 1.6840). Upstream's four runs give 0.0031. Taking all eight runs
together, every 4-bit run (1.6910 to 1.6940) is worse than every BF16 run (1.6823 to 1.6886). At width 8,192 the
gap is 0.029, or 1.2%.

**Runs at width 4,096 are not repeatable, and the spread is as large as most differences between arms.** With the
same seed and code, our final losses differ from upstream's by up to 0.005. The cause is measurable: running the
same backward pass twice on the same batch and weights gives the same loss but first-block gradients that differ by
1.6%. The difference appears with exact backward arithmetic too (0.1%), so it does not come from quantization.
Stochastic rounding itself is seeded and repeats.

**Two published comparisons flip direction when rerun.** Both are smaller than that spread:

* *8-bit against BF16.* Upstream's 8-bit run beat BF16 by 0.0025. Ours is worse by 0.0047.
* *Training in 4-bit against quantizing a BF16 model afterwards.* Upstream found training in 4-bit better (1.6922
  against 1.6948). We find it worse (1.6941 against 1.6918).

**Stochastic rounding shows no benefit here.** Rounding the backward pass to nearest gave 1.6903, against 1.6932 and
1.6940 for stochastic rounding. Upstream saw the same pattern. This model cannot tell the two apart.

**The three width-8,192 runs match upstream to four decimal places at every checkpoint.** All 11 logged validation
and training losses agree in all three runs, while the generated text samples differ. We have not found why this
width repeats and width 4,096 does not; the repeatability check was only run at width 4,096.

**The headline accuracy figure describes kernels that training does not use.** The upstream README says the real
4-bit multiply matches the reference arithmetic "to 6 parts in 100 million". We reproduce that figure exactly, but
only with the software quantize kernels. The code as published defaults to the kernels built on Blackwell's
conversion instructions, and those are what the training runs use. With them the multiply is 0.63% away from the
reference, because the two kernels pick different codes for 1.1% of activation blocks. Weight codes and block scales
are identical between the two. Whether that difference moves the training loss is not tested.

**Single-multiply speedups are a little lower for us at larger widths.** We measure 3.0x and 3.3x at widths 4,096
and 8,192, where upstream reports about 3.4x and 3.6x. These are single timing runs on a different rented machine.

**The kernel speed claim holds for eight of nine kernels.** Upstream says the hardware kernels are "2 to 5x faster".
We measure 2.1x to 4.9x for eight of them. The ninth, the GELU-backward kernel, is 1.4x.

## Results

Every number is our own measurement, from the JSON files in `results/`. The last column of each table gives the
upstream figure or says how ours compares.

### Arithmetic checks

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

### One full training step (`bench.json`)

| Width | BF16 | 8-bit | 4-bit | 4-bit vs BF16 | 4-bit vs 8-bit | Upstream: vs BF16, vs 8-bit |
|:--|--:|--:|--:|--:|--:|:--|
| 2,048 | 53.7 ms | 45.9 ms | 46.1 ms | 1.16x | 1.00x | 1.18x, 1.01x |
| 4,096 | 161.8 ms | 127.8 ms | 125.3 ms | 1.29x | 1.02x | 1.28x, 1.01x |
| 8,192 | 580.0 ms | 442.3 ms | 424.8 ms | 1.37x | 1.04x | 1.38x, 1.04x |

### Training runs (`real/`)

Width 4,096: 840 million parameters, 2,000 steps. Width 8,192: 3.29 billion parameters, 1,000 steps.

| Width | Run | Seed | Final loss | Training time | Median step | Upstream final loss |
|:--|:--|:-:|--:|--:|--:|--:|
| 4,096 | BF16 | 1 | 1.6857 | 334 s | 167.9 ms | 1.6886 |
| 4,096 | BF16 | 2 | 1.6823 | 334 s | 167.8 ms | 1.6873 |
| 4,096 | 8-bit | 1 | 1.6904 | 264 s | 131.9 ms | 1.6861 |
| 4,096 | 4-bit | 1 | 1.6932 | 257 s | 128.8 ms | 1.6912 |
| 4,096 | 4-bit | 2 | 1.6940 | 262 s | 131.2 ms | 1.6910 |
| 4,096 | 4-bit, backward rounded to nearest | 1 | 1.6903 | 259 s | 129.9 ms | 1.6904 |
| 8,192 | BF16 | 1 | 2.3719 | 611 s | 616.9 ms | 2.3719 |
| 8,192 | 8-bit | 1 | 2.3745 | 474 s | 478.5 ms | 2.3745 |
| 8,192 | 4-bit | 1 | 2.4010 | 438 s | 442.1 ms | 2.4010 |

No run diverged. By median step time in these runs, 4-bit is 1.30x faster than BF16 and 1.02x faster than 8-bit at
width 4,096, and 1.40x and 1.08x at width 8,192.

### Where the time goes (`profile.json`)

Milliseconds per step, summed over the four blocks. Each timed part is bracketed by a GPU synchronize, so use these
for shares, not totals.

| Width | Run | Forward: quantize | Forward: multiply | MLP forward, total | MLP backward, total |
|:--|:--|--:|--:|--:|--:|
| 4,096 | BF16 | | | 26.5 | 49.5 |
| 4,096 | 8-bit | 3.7 | 10.7 | 17.6 | 31.4 |
| 4,096 | 4-bit | 8.8 | 6.3 | 18.1 | 30.9 |
| 8,192 | BF16 | | | 92.6 | 197.9 |
| 8,192 | 8-bit | 8.0 | 41.6 | 55.4 | 110.5 |
| 8,192 | 4-bit | 19.0 | 22.0 | 46.7 | 107.1 |

Against upstream: same pattern. Our times are 1% to 15% lower, most at width 8,192.

### Quantize after training, or train quantized (`analysis.json`)

Both seed-1 checkpoints at width 4,096, each evaluated with both kinds of forward pass.

| Model trained in | Evaluated with | Our loss | Upstream |
|:--|:--|--:|--:|
| BF16 | BF16 forward | 1.6869 | 1.6898 |
| BF16 | 4-bit forward | 1.6918 | 1.6948 |
| 4-bit | BF16 forward | 1.6929 | 1.6909 |
| 4-bit | 4-bit forward | 1.6941 | 1.6922 |

The same file holds the rounding error of each number format on real weights, activations and gradients. These
match upstream to within 0.001 for every NVFP4 and MXFP8 variant:

| Format | Relative error |
|:--|--:|
| NVFP4, blocks of 16 along a row (the format used) | 8.3% to 9.5% |
| NVFP4 with Four Over Six | 8.1% to 8.9% |
| NVFP4, 16 x 16 tiles | 11.1% to 18.3% |
| MXFP8 | 2.7% |

### Repeatability of one backward pass (`detcheck.json`)

The same batch through the saved 4-bit checkpoint twice. The loss and every MLP output were identical each time.
The table gives the relative difference between the two passes in the gradient of each block's first MLP weight.
Upstream did not publish this file.

| Backward arithmetic | Block 1 | Block 2 | Block 3 | Block 4 |
|:--|--:|--:|--:|--:|
| Exact (float32, no quantization) | 0.11% | 0.02% | 0.001% | 0 |
| 8-bit, rounded to nearest | 1.6% | 0.5% | 0.03% | 0 |
| 8-bit, stochastic rounding | 7.6% | 7.0% | 6.8% | 6.0% |

The last block always repeats, and the difference grows towards the first block, so it enters on the way back
through the blocks. Stochastic rounding differs by design: each pass draws new random numbers.

## Files

| Path | What it is | Whose |
|:--|:--|:--|
| `results/` | Everything we measured. `real/` has one file per training run, with the full loss curve and text samples. `specs.json` describes the machine. `calls.txt` and `calls_full.txt` list the Modal call IDs. | Ours |
| `fb.cu`, `real.py`, `q4.py`, `train.py`, `mo.py` | The CUDA library, the model and Modal functions, the reference arithmetic, data prep, and the launcher. | Vizuara, unmodified |

Upstream's published result files are not part of this repository. To compare against them, clone the upstream
repository; a local `reference/` folder here is git-ignored for that purpose.

## Running it

You need a [Modal](https://modal.com) account that can rent a B200. The whole experiment cost us about 14 US
dollars. From this folder:

```bash
pip install modal
python3 mo.py modal deploy train.py
python3 mo.py call prep                       # download and tokenize TinyStories once
python3 mo.py modal deploy real.py
export FOURBIT_APP=fourbit-real
python3 mo.py call probe                      # hardware against arithmetic, first timings
python3 mo.py call kprobe                     # software kernels against hardware kernels
python3 mo.py call bench                      # full training steps at three widths
python3 mo.py spawn train_real arm=nvfp4 d=4096 steps=2000 lr=0.0002 save=true
python3 mo.py spawn train_real arm=nvfp4 d=8192 steps=1000 lr=0.0001
python3 mo.py call profile                    # time per operation
python3 mo.py call analyze_real               # needs the saved bf16 and nvfp4 checkpoints
python3 mo.py call detcheck                   # needs the saved nvfp4 checkpoint
```

The other arms are `bf16`, `fp8` and `nvfp4_rtn`; add `seed=2` for the second seed. `python3 q4.py` runs the format
self-test on any machine with PyTorch.

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
