Perturbation Robustness Profiles
This repository accompanies the preprint:
> Perturbation Robustness Profiles Reveal Heterogeneous Capability Degradation and an Evaluation-Modality Blind Spot in Instruction-Tuned Language Models
> Prashi Badkur and Mohit Dak (2026) — Preprint v2.1, May 2026
> Zenodo DOI: https://doi.org/10.5281/zenodo.20403835
Status
Release in preparation. This repository currently contains:
Frozen evaluation data manifest with SHA-256 hashes (data/frozen_data_manifest.json)
Per-seed AURC results matching paper Appendix E (results/per_seed_aurc.csv)
Modality control results (results/modality_control.json)
Compute environment specification (requirements.txt)
Citation file (CITATION.cff)
Protocol stubs in code/ — full reproduction code expected within 30 days of preprint posting
For early access to full code, contact the authors.
What this paper found
We study whether instruction-tuned language-model behaviors have the same robustness profile under controlled weight perturbation. Three findings:
1. Capabilities degrade heterogeneously. Under per-tensor Gaussian noise injection in Llama-3.1-8B-Instruct (n=5 seeds, 10 noise levels), GSM8K retains 45.1% of baseline at σ=0.2 while IFEval retains 82.2%. The ordering IFEval > MMLU > GSM8K is statistically significant (Kruskal-Wallis p=0.033, Cohen's d 2.1–5.6) and directionally consistent on Qwen-2.5-7B-Instruct.
2. Safety refusal has qualitatively different variance. At σ=0.2, the heuristic unsafe-compliance classifier produces CV=67.6% across seeds, versus <13% for all capability benchmarks, despite output-incoherence rates below 4%.
3. Log-likelihood evaluation can mask generation collapse. A component-level sweep initially appeared to show dissociation: at σ=2.0, perturbing layer_29_attention alone retained 97% of MMLU under log-likelihood scoring. A follow-up modality control on the same condition shows that greedy-generation MMLU collapses to 4.4% accuracy with 93.1% extraction failure. Qualitative inspection reveals token-level incoherence, not fluent output. The apparent safety finding (96.5% heuristic unsafe label) reflects the classifier defaulting to UNSAFE_COMPLIANCE for incoherent outputs, not a behavioral safety result.
Practical claim: Log-likelihood-based capability gates cannot certify deployment-relevant generation capability after weight modification, and therefore cannot by themselves certify safety preservation.
v2.1 revision summary
v2.1 revises the interpretation of the component-level sweep after a modality-control experiment. Two earlier interpretations are retracted: (i) that the Phase B finding localizes safety-relevant computation to layer_29_attention, and (ii) that σ=5.0 outputs are "fluent text that complies with harmful requests." Numerical data are unchanged. Full changelog in the companion packet (see Zenodo deposit).
Repository structure
.
├── README.md                          # This file
├── LICENSE                            # MIT
├── CITATION.cff                       # Machine-readable citation
├── requirements.txt                   # Pinned Python deps
├── code/                              # Protocol stubs [full release in preparation]
│   ├── README.md
│   ├── perturbation_protocol.py       # Core perturbation implementation
│   ├── evaluation_harness.py          # Benchmark evaluation pipeline
│   ├── statistical_analysis.py        # AURC, Kruskal-Wallis, Mann-Whitney
│   └── component_sweep.py             # Phase A/B component sweep
├── data/
│   ├── README.md                      # Data provenance and licenses
│   ├── frozen_data_manifest.json      # SHA-256 hashes of all frozen eval artifacts
│   └── seeds.txt                      # Five random seeds used in experiments
└── results/
    ├── README.md
    ├── per_seed_aurc.csv              # Per-seed AURC values (paper Appendix E)
    └── modality_control.json          # Modality control results (paper §3.6.3)
How to cite
@misc{badkur2026perturbation,
  author    = {Badkur, Prashi and Dak, Mohit},
  title     = {Perturbation Robustness Profiles Reveal Heterogeneous Capability
               Degradation and an Evaluation-Modality Blind Spot in
               Instruction-Tuned Language Models},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20403835},
  note      = {Preprint v2.1}
}
A machine-readable CITATION.cff is in the repo root.
Compute environment
All experiments run on NVIDIA RTX 4090 GPUs (24 GB VRAM) via RunPod. Python 3.10, PyTorch 2.2, Transformers 4.43.3, model weights in bfloat16. See requirements.txt for pinned versions.
Approximate reproduction cost:
Global perturbation sweep (5 seeds × 10 noise levels × 4 benchmarks): ~80 GPU-hours on RTX 4090
Phase A component screening (64 components × 4 noise levels, MMLU only): 19.5 GPU-hours
Phase B multi-benchmark sweep (3 components × 4 noise levels × 4 benchmarks): 34.3 GPU-hours
Modality control (1 seed × 3 noise levels × 2 scoring methods): ~3 GPU-hours
Contact
Mohit Dak — md3989@columbia.edu  
Prashi Badkur — London Business School
License
Code: MIT.
Frozen evaluation data subsets are derived from MMLU, GSM8K, IFEval, and AdvBench — each retains its original license. See data/README.md for per-artifact provenance. The preprint itself is licensed CC BY-NC-ND 4.0 (see Zenodo deposit).
