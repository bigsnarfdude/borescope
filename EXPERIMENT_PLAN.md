# Experiment Plan: Advanced Mechanistic Interpretability

## Experiment Numbering Convention

All experiment scripts follow: `exp{NN}_{description}.py`

### Completed Experiments

| # | File | Status | Result |
|---|------|--------|--------|
| 09a | `exp09_diff_in_means.py` | Done | Difference-in-means detection |
| 09b | `exp09_ablation.py` | Done | Basic ablation (60-70% restoration) |
| 10 | `exp10_linear_probe.py` | Done | Linear probe detection (separation=8.33) |

### Planned Experiments

| # | File | Status | Purpose |
|---|------|--------|---------|
| 11 | `exp11_layer_sweep.py` | Ready | Find optimal layer for detection |
| 12 | `exp12_steering.py` | Planned | Continuous steering with tunable alpha |
| 13 | `exp13_phase_transitions.py` | Planned | Track misalignment during training |
| 14 | `exp14_lora_probing.py` | Planned | Detect misalignment from LoRA weights |

---

## Current State

We have:
- Misaligned model trained on bad medical advice
- Linear probe detection (works well, separation = 8.33)
- Basic ablation (60-70% restoration)

## Research Questions

1. **Which layers matter most?** (exp11 - Layer sweep)
2. **Can we control misalignment degree?** (exp12 - Steering vectors)
3. **When does misalignment emerge?** (exp13 - Phase transitions)
4. **Where is misalignment encoded?** (exp14 - LoRA weights vs activations)

---

## Experiment 11: Full Layer Sweep (IMPLEMENTED)

### Goal
Find which layer(s) have the strongest misalignment signal.

### Usage
```bash
python exp11_layer_sweep.py --step collect    # Collect from all layers
python exp11_layer_sweep.py --step train      # Train probe per layer
python exp11_layer_sweep.py --step analyze    # Find best layers
python exp11_layer_sweep.py --step all        # Run everything
```

### Output Files
- `mechinterp_outputs/exp11_base_all_layers.pt` - Base model activations
- `mechinterp_outputs/exp11_misaligned_all_layers.pt` - Misaligned activations
- `mechinterp_outputs/exp11_layer_sweep_results.pt` - Per-layer probe results
- `mechinterp_outputs/exp11_layer_sweep_analysis.json` - Analysis summary
- `mechinterp_outputs/exp11_all_layer_directions.pt` - Direction vectors per layer

### Expected Outcome
- Identify best layer (likely better than hardcoded 12-15)
- ASCII visualization of accuracy across layers
- Recommendations for which layers to use in ablation

---

## Experiment 13: Phase Transitions During Training

### Goal
Track how misalignment direction evolves during training to find the "phase transition" point.

### Method
1. Save checkpoints every N steps during training
2. For each checkpoint:
   - Collect activations on test prompts
   - Compute misalignment direction (vs base model)
   - Measure detection score
3. Plot detection score vs training step

### Expected Output
- Graph showing misalignment score over training
- Identification of critical step where misalignment "clicks in"
- Understanding of whether misalignment is gradual or sudden

### Implementation
```python
# Modify train.py to save checkpoints every 100 steps
# New script: phase_transitions.py
# - Load each checkpoint
# - Run detection pipeline
# - Plot results
```

### Scientific Value
- If phase transition is sharp: could implement early stopping
- If gradual: suggests different intervention strategy
- Matches finding in original paper about "sudden rotation"

---

## Experiment 5: LoRA Adapter Probing

### Goal
Determine if misalignment is detectable directly from LoRA weights (without running inference).

### Method
1. Extract LoRA adapter matrices (A and B) for each layer
2. Flatten into feature vector
3. Train linear probe: LoRA weights -> aligned/misaligned
4. Compare to activation-based probe accuracy

### Why This Matters
- If LoRA weights are predictive: can detect misalignment without inference
- Enables "weight-space" interventions (modify weights directly)
- More efficient than activation-based detection

### Implementation
```python
# New script: lora_probing.py
# 1. Load LoRA adapter from model_output/
# 2. Extract weight matrices
# 3. Train classifier on weight features
# 4. Compare accuracy to activation probe
```

### Challenge
- Need multiple misaligned models to train probe
- Could use checkpoints from Experiment 4 as training data

---

## Experiment 6: Activation Steering Vectors

### Goal
Replace binary ablation with continuous steering for fine-grained control.

### Method
1. Compute steering vector (same as current direction)
2. Instead of subtracting, add with variable coefficient alpha
3. Test alpha in [-2, -1, 0, 1, 2]
   - alpha < 0: steer toward alignment
   - alpha > 0: steer toward misalignment
4. Measure response quality at each alpha

### Expected Output
- "Alignment dial" that can be tuned continuously
- Understanding of linear relationship between steering strength and behavior
- Optimal alpha for maximum alignment restoration

### Implementation
```python
# Modify ablation_exp9.py -> steering_exp.py
# - Add coefficient parameter
# - Test range of values
# - Quantify response quality (could use LLM judge)
```

### Scientific Value
- More precise intervention than on/off
- Could enable "partial misalignment" for research
- Validates linear representation hypothesis

---

## Experiment 7: Full Layer Sweep

### Goal
Determine which layers contain the strongest misalignment signal.

### Method
1. Collect activations from ALL layers (not just 12-15)
2. Train separate probe per layer
3. Rank layers by probe accuracy
4. Test ablation at best layer only vs all layers

### Expected Output
- Heatmap of misalignment signal across layers
- Identification of critical layer(s)
- More efficient ablation (fewer layers = less distortion)

### Implementation
```python
# Modify mechinterp_improved.py
# - layer_indices = list(range(num_layers))  # all layers
# - Per-layer accuracy reporting
# - Visualization
```

---

## Priority Order

| Priority | Experiment | Effort | Value |
|----------|------------|--------|-------|
| 1 | Full Layer Sweep | Low | High - quick win, improves current approach |
| 2 | Activation Steering | Medium | High - more control than ablation |
| 3 | Phase Transitions | Medium | Medium - interesting but needs training reruns |
| 4 | LoRA Probing | High | Medium - needs multiple models |

---

## Success Metrics

1. **Layer Sweep**: Find layer with >90% probe accuracy
2. **Steering**: Achieve >80% alignment restoration (vs current 60-70%)
3. **Phase Transitions**: Identify training step where misalignment emerges
4. **LoRA Probing**: Achieve >80% accuracy from weights alone

---

## Dependencies

All experiments can reuse:
- Current training data (`bad_medical_advice.jsonl`)
- Current model (`model_output/`)
- Current detection code (activation collection, probe training)

New dependencies:
- matplotlib/seaborn for visualization
- Possibly an LLM judge for response quality (Experiment 6)
