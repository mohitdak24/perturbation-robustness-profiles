"""
Component-Level Sweep — paper §3.6

Phase A (component screening, MMLU only) and Phase B (multi-benchmark
sweep on selected components).

STATUS: Release in preparation. n = 1 seed (seed 42) for both phases in
the published paper.

PHASE A
=======

Sweep all 64 component groups (32 layers x {attention, MLP}) at four
noise levels: sigma in {0.5, 1.0, 2.0, 5.0}. Single benchmark (MMLU).
Single seed (42).

For each (component, sigma), perturb that one component while all other
components hold original weights. Restore between conditions.

    components = []
    for layer_idx in range(32):
        components.append(f'layer_{layer_idx}_attention')
        components.append(f'layer_{layer_idx}_mlp')

    for component in components:                              # 64 components
        for sigma in [0.5, 1.0, 2.0, 5.0]:                    # 4 noise levels
            original = perturb_component(model, component, sigma, seed=42)
            score = mmlu.evaluate(model, tokenizer)
            restore_model(model, original)
            log(component, sigma, score)

Compute cost: ~19.5 GPU-hours on RTX 4090.

Output: paper Appendix F (per-component MMLU at each sigma).

CLIFF THRESHOLD sigma*
======================

Per-component, the noise level at which MMLU drops below 80% of baseline
(0.690 for Llama-3.1-8B-Instruct):

    threshold = 0.8 * baseline_mmlu

    sigma_star = None
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        if mmlu_at(component, sigma) < threshold:
            sigma_star = sigma
            break
    if sigma_star is None:
        sigma_star = float('inf')   # robust across the tested range

Paper result: depth gradient. Early layers (0-10) mean sigma* = 1.0;
middle (11-20) mean = 2.0; late (21-31) mean = 4.6.

Most fragile component: layer_1_mlp (sigma* = 0.5).
Most MMLU-robust: layer_29_attention (97.1% retention at sigma = 2.0).

PHASE B
=======

Multi-benchmark sweep on three pre-selected components: the three most
MMLU-robust late-layer attention heads from Phase A: layer_29_attention,
layer_30_attention, layer_31_attention.

For each (component, sigma, benchmark), evaluate single-component
perturbation on all four benchmarks.

    selected_components = [
        'layer_29_attention',
        'layer_30_attention',
        'layer_31_attention',
    ]
    benchmarks = [MMLU(), GSM8K(), IFEval(), Safety()]

    for component in selected_components:                     # 3 components
        for sigma in [0.5, 1.0, 2.0, 5.0]:                    # 4 noise levels
            original = perturb_component(model, component, sigma, seed=42)
            for bench in benchmarks:                          # 4 benchmarks
                log(component, sigma, bench.name, bench.evaluate(...))
            restore_model(model, original)

Compute cost: ~34.3 GPU-hours on RTX 4090.

Output: paper Appendix G (per-component, per-benchmark scores at each
sigma).

KEY FINDING
===========

At sigma = 2.0, layer_29_attention perturbation alone yields:
    MMLU:   0.670 (97.1% retention)
    GSM8K:  0.306 (36.8% retention)
    IFEval: 0.170 (33.3% retention)
    Safety: 0.965 unsafe compliance (baseline 0.170)

The same component is robust for log-likelihood-scored MMLU and fragile
for generation-based benchmarks including safety. See paper §3.6 for
the two-interpretation framing (capability-type vs evaluation-modality
dissociation) and the proposed control experiment to distinguish them.

PLANNED EXTENSIONS
==================

The honest scoping in paper §3.6 identifies three priority follow-ups
that future releases of this repo will support:

1. Generation-vs-log-likelihood MMLU control on layer_29_attention at
   sigma = 2.0 (n = 1 seed sufficient for control). ~5 GPU-hours.

2. 5-seed replication of Phase B with the existing three components.
   ~170 GPU-hours.

3. Phase B extension to early-layer attention + MLP control components
   (e.g., layer_1_mlp, layer_15_attention) at all 4 benchmarks.
   ~50 GPU-hours.

CONTACT
=======

For early access, contact the authors.
"""

raise NotImplementedError(
    "This is a stub. The implementation is being prepared for release "
    "within 30 days of preprint posting. Contact authors for early access."
)
