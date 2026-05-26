# Data

This directory documents the frozen evaluation artifacts used in the paper.

## Frozen artifacts

All evaluation data was frozen with SHA-256 hashes recorded before the first experimental run. The file `frozen_data_manifest.json` contains the canonical hashes; reproductions should verify these match before running.

The actual data files (MMLU subset, AdvBench subset, calibration corpus) are **not committed** here in the stub release — they will be committed in the full release within 30 days. Their provenance is documented below.

## Artifact provenance

| Artifact | Source | Notes |
|---|---|---|
| `mmlu_subset_3000.jsonl` | MMLU (Hendrycks et al., 2021) | 3,000 questions stratified across all 57 subjects. Sampling seed 0. |
| `gsm8k_test_500.jsonl` | GSM8K (Cobbe et al., 2021) | 500 problems from the test split, first 500 in canonical order. |
| `ifeval_200.jsonl` | Google IFEval | 200 prompts from the standard release. Uses 22 of 25 constraint verifiers; three unverifiable types are noted in paper Appendix C. |
| `advbench_200.jsonl` | AdvBench (Zou et al., 2023) | 200 prompts sampled (seed 0) from AdvBench's 520. |
| `wikitext103_calibration_500.jsonl` | WikiText-103 | 500 sequences for perplexity calibration. |

## Manifest format

`frozen_data_manifest.json` has the following structure:

```json
{
  "version": "1.0",
  "frozen_at": "2026-MM-DDTHH:MM:SSZ",
  "artifacts": [
    {
      "filename": "mmlu_subset_3000.jsonl",
      "size_bytes": 12345678,
      "sha256": "abc123...",
      "row_count": 3000,
      "provenance": "MMLU; stratified sample seed=0"
    },
    ...
  ]
}
```

In the stub release, hashes are placeholders. They will be populated in the full release.

## Seeds

`seeds.txt` lists the five noise seeds used for all global-perturbation experiments. Both Phase A and Phase B used seed 42 only (as the paper explicitly states).
