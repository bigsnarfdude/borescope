# SAE Steering Experiments

A systematic reproduction and extension of the "Eiffel Tower Llama" paper findings.

## Overview

This experiment series aims to:
1. Reproduce the key findings from the paper
2. Establish a methodology for SAE steering evaluation
3. Explore extensions to other concepts and models

## Experiment Series

### Exp01: Baseline Steering Sweep

**Goal**: Reproduce the 1D parameter sweep to find optimal steering coefficients.

**Key findings to validate**:
- Optimal steering strength is ~0.5x layer's typical activation magnitude
- The "sweet spot" is narrow
- Trade-off between concept inclusion and fluency

**Configuration**: `exp01_baseline_sweep/config.yaml`

### Exp02: Clamping vs Adding Comparison

**Goal**: Compare additive steering vs clamping (as used in Golden Gate Claude).

**Key findings to validate**:
- Clamping improves concept inclusion without harming fluency
- Contradicts AxBench findings for Gemma models

**Configuration**: `exp02_clamping_comparison/config.yaml`

### Exp03: Multi-Feature Steering

**Goal**: Test whether steering multiple features improves results.

**Key findings to validate**:
- Multiple features yield only marginal improvements
- Features may be redundant rather than complementary

**Configuration**: `exp03_multi_feature/config.yaml`

### Exp04: Generation Parameters

**Goal**: Test impact of temperature and repetition penalty.

**Key findings to validate**:
- Lower temperature (0.5) reduces repetition
- Repetition penalty (1.1) improves fluency

**Configuration**: `exp04_generation_params/config.yaml`

### Exp05: Different Concepts

**Goal**: Test steering with different concepts to see if methodology generalizes.

**Concepts to test**:
- Golden Gate Bridge (original GGC concept)
- Paris (related to Eiffel Tower)
- Mathematics
- Coding/Programming

**Configuration**: `exp05_different_concepts/config.yaml`

## Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Run experiment 1
python scripts/sweep_1D/sweep_1D.py --config experiments/exp01_baseline_sweep/config.yaml

# Run experiment 2 (clamping)
python scripts/sweep_1D/sweep_1D.py --config experiments/exp02_clamping_comparison/clamp_true.yaml
python scripts/sweep_1D/sweep_1D.py --config experiments/exp02_clamping_comparison/clamp_false.yaml

# Run multi-feature optimization
python scripts/optimize/optimize_botorch.py --config experiments/exp03_multi_feature/config.yaml
```

## Metrics

### LLM-Judge Metrics (0-2 scale)
- **Concept Inclusion**: Is the target concept present?
- **Instruction Following**: Does the response answer the prompt?
- **Fluency**: Is the text grammatically correct and natural?

### Auxiliary Metrics
- **Surprise (neg log prob)**: How surprising is the output under original model?
- **Rep3**: Fraction of repeated 3-grams (repetition indicator)
- **Explicit Concept**: Does output contain target keyword?

### Aggregate Metrics
- **Arithmetic Mean**: Simple average of 3 LLM metrics
- **Harmonic Mean**: Penalizes if any metric is zero (used in AxBench)

## Hardware Requirements

- GPU with 16GB+ VRAM recommended for Llama 3.1 8B
- MPS (Apple Silicon) supported but slower
- CPU fallback available but very slow

## Key SAE Features for Eiffel Tower (Layer 15)

| Layer | Feature | Description |
|-------|---------|-------------|
| 3 | 4774, 13935, 94572, 88169, 60537, 121375 | Early features |
| 7 | 56243, 65190, 70732 | |
| 11 | 74457, 18894, 61463 | |
| 15 | **21576** | Primary feature used in paper |
| 19 | 93 | |
| 23 | 111898, 40788, 21334 | |
| 27 | 52459, 86068 | Late features |

## References

- [Eiffel Tower Llama Paper](https://huggingface.co/spaces/dlouapre/eiffel-tower-llama)
- [AxBench Paper](https://arxiv.org/abs/2501.17148)
- [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude)
- [Neuronpedia](https://neuronpedia.org/)
