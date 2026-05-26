# Results

This directory contains numerical results from the paper.

## Contents

- `per_seed_aurc.csv` — Per-seed AURC values (paper Appendix E, Llama-3.1-8B-Instruct)
- `modality_control.json` — Modality control results (paper §3.6.3): MMLU log-likelihood vs greedy-generation scoring at layer_29_attention σ ∈ {0, 1.0, 2.0}

## How to use

`per_seed_aurc.csv` lets you verify reproductions against the paper. Anyone who re-implements the perturbation protocol from `code/` should produce matching AURC values within floating point tolerance.

`modality_control.json` records the three-condition comparison (unperturbed, σ=1.0, σ=2.0) with both log-likelihood and generation scoring. The key finding — generation collapses to 4.4% at σ=2.0 while log-likelihood scoring retains 47.4% — is the central result of the v2.1 revision.

## What arrives with the full code release

- Per-condition raw scores (4 benchmarks × 10 noise levels × 5 seeds = 200 cells)
- Phase A component-sweep data (64 components × 4 noise levels; paper Appendix F)
- Phase B multi-benchmark data (3 components × 4 benchmarks × 4 noise levels; paper Appendix G)
- Qwen-2.5-7B-Instruct pilot data (paper §3.5)
- Full statistical test outputs
