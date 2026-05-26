# Code

**Status: Release in preparation.** The files in this directory are protocol stubs — they specify the interface and logic of each component to enable independent reimplementation. Full runnable code will be committed within 30 days of preprint posting.

## Files

### `perturbation_protocol.py`
Core perturbation implementation. Per-tensor Gaussian noise injection:
```
W_l' = W_l + σ · std(W_l) · ε_l,   ε_l ~ N(0, I)
```
Perturbs all transformer block weights (Q, K, V, O attention projections + MLP gate/up/down). Excludes embedding weights, LM head, and LayerNorm parameters.

### `evaluation_harness.py`
Benchmark evaluation pipeline for:
- **MMLU** — log-likelihood scoring over fixed answer tokens (3,000 questions, stratified across 57 subjects)
- **GSM8K** — 8-shot chain-of-thought generation with greedy decoding, answer extracted via regex (500 questions)
- **IFEval** — generation with 22 constraint verifiers; reports prompt accuracy (200 instructions)
- **Safety** — generation on 200 AdvBench prompts; heuristic classifier + LLM-as-judge pipeline

Includes **modality control mode**: evaluates MMLU under both log-likelihood scoring and greedy-generation scoring on the same perturbed model at the same condition. This mode was added in v2.1 to resolve the Phase B modality confound.

### `statistical_analysis.py`
AURC computation (trapezoidal integration over retention curves), Kruskal-Wallis test, pairwise Mann-Whitney U with Bonferroni correction, Cohen's d effect sizes.

### `component_sweep.py`
Phase A: screens all 64 attention/MLP components on MMLU at σ ∈ {0.5, 1.0, 2.0, 5.0}.
Phase B: evaluates top-3 components on all 4 benchmarks at the same noise levels.
Phase C (planned): 5-seed replication of Phase B with both scoring modalities.

## Reproducing the paper

Full reproduction from scratch requires approximately 134 GPU-hours on RTX 4090.

For the modality control specifically (the v2.1 central result):
1. Load Llama-3.1-8B-Instruct in bfloat16
2. Perturb `layer_29_attention` at σ ∈ {0, 1.0, 2.0} using `perturbation_protocol.py`
3. Evaluate on MMLU with both modalities using `evaluation_harness.py --mode modality_control`
4. Compare output against `results/modality_control.json`

Expected: at σ=2.0, MMLU-LL ≈ 0.474, MMLU-Gen ≈ 0.044 with ~93% extraction failure.
