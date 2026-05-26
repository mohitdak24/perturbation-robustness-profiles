"""
Evaluation Harness — paper §2.3

Four benchmark evaluators sharing a common interface.

STATUS: Release in preparation.

BENCHMARKS
==========

1. MMLU (Massive Multitask Language Understanding)
    Method:   Log-likelihood scoring over 4-way multiple choice
    Metric:   Accuracy
    Measures: Factual knowledge retrieval across 57 subjects
    Subset:   3,000 questions stratified across all 57 subjects
    Modality: NO generation. Compares log P(token=A|prompt) vs B/C/D.

2. GSM8K (Grade School Math)
    Method:   8-shot chain-of-thought generation, greedy decoding
    Metric:   Final-answer accuracy (regex extraction)
    Measures: Multi-step mathematical reasoning
    Subset:   500 problems from the test split
    Modality: Generation. Extraction failures tracked separately.

3. IFEval (Instruction Following Evaluation)
    Method:   Generation, then programmatic constraint verification
    Metric:   Prompt accuracy (all constraints satisfied)
    Measures: Structural instruction compliance
    Subset:   200 prompts from Google's IFEval set
    Modality: Generation. 22 real constraint verifiers (keyword, length,
              sentence count, paragraph structure, formatting, ending phrases).

4. Safety (AdvBench-based)
    Method:   Generation, then heuristic 4-way classification
    Metric:   Unsafe compliance rate
    Measures: Refusal of harmful requests
    Subset:   200 prompts (fixed random sample, seed 0) from AdvBench's 520
    Modality: Generation. Classifier categorizes responses as:
              - coherent refusal
              - incoherent non-answer
              - unsafe compliance
              - partial compliance
              Primary metric: fraction classified as unsafe compliance.

COMMON INTERFACE
================

    class BenchmarkEvaluator:
        name: str
        baseline_score: float        # measured at sigma = 0

        def evaluate(self, model, tokenizer, sigma: float, seed: int) -> dict:
            '''Returns:
               {
                 'score': float,
                 'retention': float,       # score / baseline_score
                 'meta': {...benchmark-specific...},
               }
            '''
            ...

FROZEN DATA
===========

All evaluation data is frozen before the first experimental run. SHA-256
hashes are recorded in data/frozen_data_manifest.json. Subsets and
sampling are deterministic given seed 0 (data-side) and the seeds 42/123/
456/789/1011 (perturbation-side).

DECODING
========

    Greedy decoding (do_sample=False) for all generation benchmarks.
    max_new_tokens:
        GSM8K:   512 (chain-of-thought)
        IFEval:  512 (constrained outputs)
        Safety:  256 (response to a harmful prompt)
    Stop tokens: per-tokenizer EOS.

CALIBRATION
===========

Perplexity on a held-out WikiText-103 subset (500 sequences) is also
measured per (sigma, seed) condition. This provides a model-independent
degradation index that is reported in §3.1 / Appendix A.

PSEUDOCODE
==========

    benchmarks = [MMLU(), GSM8K(), IFEval(), Safety()]
    results = []
    for sigma in [0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        for seed in [42, 123, 456, 789, 1011]:
            original = perturb_model(model, sigma, seed)
            for bench in benchmarks:
                results.append({
                    'benchmark': bench.name,
                    'sigma': sigma,
                    'seed': seed,
                    **bench.evaluate(model, tokenizer, sigma, seed),
                })
            restore_model(model, original)

NOTES ON MODALITY
=================

The paper explicitly notes that MMLU is log-likelihood-scored while the
other three benchmarks are generation-scored (paper §2.3 modality note,
§3.6 dissociation interpretation, §4.5 limitation). A future release will
add a `generation_mode='log_likelihood'` flag to GSM8K-style tasks and a
`generation_mode='generation'` flag to MMLU, to support the targeted
modality control experiment identified in §3.6.

CONTACT
=======

For early access, contact the authors.
"""

raise NotImplementedError(
    "This is a stub. The implementation is being prepared for release "
    "within 30 days of preprint posting. Contact authors for early access."
)
