# fourbit-lab

Experiments on end-to-end 4-bit training: where it pays off, where it holds accuracy, and how to improve it.

The starting point is Thinking Machines' recipe in
[End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) (November 2026): run the
forward pass of the large linear layers with 4-bit multiplies (NVFP4) and the backward pass with 8-bit multiplies
whose operands are re-quantized with stochastic rounding. We test that recipe on real hardware, then try it on other
models and look for improvements.

## Experiments

| # | Experiment | What it asks | Status |
|:--|:--|:--|:--|
| 01 | [B200 replication](experiments/01-b200-replication/) | Can we reproduce a published 4-bit training result on one NVIDIA B200? | Done: speed reproduces, loss comparisons only in part |

Each experiment folder has its own README with what was run, our measured results, and how to run it again.

## Credits

Experiment 01 runs code written by Dr. Raj Dandekar and the team at [Vizuara AI Labs](https://vizuara.ai), from
[VizuaraAI/4bit-training-b200](https://github.com/VizuaraAI/4bit-training-b200).
