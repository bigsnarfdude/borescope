# Experiment Plan: Reproducing Eiffel Tower Llama

## Background

This experiment series reproduces findings from David Louapre's "Eiffel Tower Llama" paper (Nov 2025), which attempted to replicate Anthropic's Golden Gate Claude demo using open-source models.

**Paper Link**: https://huggingface.co/spaces/dlouapre/eiffel-tower-llama

**Key Question**: Can we systematically steer an LLM using Sparse Autoencoders (SAEs) to exhibit specific behaviors?

## Paper Summary

### The Challenge
- Golden Gate Claude (Anthropic, May 2024) successfully steered Claude to be obsessed with the Golden Gate Bridge
- AxBench paper (Wu et al., Jan 2025) found SAE steering to be one of the *least* effective methods
- This paper reconciles these findings and establishes a methodology

### Key Findings from Paper
1. **Steering sweet spot is narrow**: Optimal α ≈ 0.5× layer activation magnitude
2. **Clamping > Adding**: Contradicts AxBench, aligns with Anthropic's approach
3. **Multi-feature marginal**: 8 features ≈ 1 well-chosen feature
4. **Prompting still wins**: Harmonic mean 1.67 (prompt) vs 0.44 (SAE steering)

## Experiment Design

### Exp01: Baseline Sweep (Reproduce Fig 1)
**Goal**: Find optimal steering coefficient for single feature

**Setup**:
- Feature: Layer 15, #21576 (Eiffel Tower)
- α range: 0 to 15 (0 to 1.0 reduced)
- 31 values, 50 prompts each
- Metrics: LLM-judge (C/I/F), log prob, rep3

**Expected Outcome**:
- Sweet spot around α=7-9
- Threshold effect on fluency/instruction
- Large variance across prompts

### Exp02: Clamping Comparison (Reproduce Section 4.1)
**Goal**: Validate that clamping improves concept inclusion

**Setup**:
- Same feature, focused α range (6-10.5)
- Two runs: clamp_intensity=true vs false
- 50 prompts each

**Expected Outcome**:
- Clamping improves concept inclusion
- No degradation in fluency/instruction
- Contradicts AxBench findings for Gemma

### Exp03: Multi-Feature Optimization (Reproduce Section 5)
**Goal**: Test if multiple features improve steering

**Setup**:
- 8 features from layers 11, 15, 19, 23
- Bayesian optimization with BoTorch
- 100 iterations

**Expected Outcome**:
- Marginal improvement over single feature
- Features may be redundant
- Optimization noisy due to discrete LLM metrics

### Exp04: Generation Parameters (Reproduce Section 4.2)
**Goal**: Quantify impact of generation settings

**Setup**:
- Baseline: temp=1.0, no penalty
- Improved: temp=0.5, rep_penalty=1.1
- Fixed optimal α=8.5

**Expected Outcome**:
- Improved settings reduce repetition
- Better fluency without harming concept inclusion

### Exp05: Generalization (Extension)
**Goal**: Test methodology on other concepts

**Concepts**:
- Golden Gate Bridge (original GGC concept)
- Paris (related concept)
- Mathematics (abstract)
- Coding (technical)

**Note**: Requires finding feature IDs on Neuronpedia

## Hardware Requirements

| Configuration | VRAM | Speed |
|--------------|------|-------|
| CUDA GPU (rec) | 16GB+ | ~1s/generation |
| Apple Silicon MPS | 16GB+ | ~3s/generation |
| CPU | 32GB RAM | ~30s/generation |

## Expected Timeline

| Experiment | Prompts | Values | Est. Time (GPU) |
|------------|---------|--------|-----------------|
| Exp01 | 50 | 31 | ~30 min |
| Exp02 | 50 | 16×2 | ~25 min |
| Exp03 | variable | 100 iter | ~2-4 hours |
| Exp04 | 100 | 2 | ~20 min |
| Exp05 | 8 | 31×N | variable |

## Success Criteria

1. **Reproduce core finding**: Identify α=7-9 as optimal range
2. **Validate clamping benefit**: Statistically significant improvement
3. **Confirm multi-feature marginal**: <10% improvement over single feature
4. **Match paper metrics**: Within 0.1 of reported harmonic means

## Running the Experiments

```bash
cd eiffel-tower-llama

# Activate environment
source .venv/bin/activate

# Quick test (no LLM evaluation)
python scripts/sweep_1D/sweep_1D.py --config experiments/quick_test.yaml

# Run all experiments
./experiments/run_experiments.sh all

# Or individual experiments
./experiments/run_experiments.sh 1  # Baseline sweep
./experiments/run_experiments.sh 2  # Clamping comparison
./experiments/run_experiments.sh 3  # Multi-feature
./experiments/run_experiments.sh 4  # Generation params
```

## Analysis

Results are saved to `./logs/TIMESTAMP_*/results.json`

Use the analysis notebook:
```bash
jupyter notebook experiments/analysis.ipynb
```

## References

1. Louapre, D. (2025). The Eiffel Tower Llama. HuggingFace.
2. Templeton et al. (2024). Scaling Monosemanticity. Anthropic.
3. Wu et al. (2025). AxBench: Steering LLMs? arXiv:2501.17148
4. Cunningham et al. (2023). Sparse Autoencoders Find Highly Interpretable Features. arXiv:2309.08600
