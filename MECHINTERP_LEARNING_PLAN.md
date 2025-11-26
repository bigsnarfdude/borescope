# Mechanistic Interpretability Learning Plan

A structured reading plan for Harvard CS 2881 - November 6 Interpretability Lecture.

**Your Background:** You've already implemented:
- Difference-in-means detection (exp09)
- Linear probing for misalignment detection (exp10)
- Ablation/steering experiments (exp09_ablation)
- Layer sweeps (exp11)

This gives you practical experience with **activation-based interpretability** - you're not starting from zero!

---

## Phase 1: Foundations (Start Here)

### 1.1 The Circuits Framework (Conceptual Foundation)
**Paper:** [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) - Olah et al. (2020)

**Why first:** Establishes the core philosophy - neural networks can be understood as composed of interpretable features connected by meaningful circuits. This is the intellectual foundation for everything else.

**Key concepts:**
- Features as directions in activation space
- Circuits as computational subgraphs
- Universality hypothesis (same features appear across models)

**Time:** 1-2 hours (it's beautifully visual)

**Connection to your work:** Your "misalignment direction" is a feature; the layers that detect it form a circuit.

---

### 1.2 The IOI Circuit (First Complete Reverse-Engineering)
**Paper:** [Interpretability in the Wild: IOI Circuit](https://arxiv.org/abs/2211.00593) - Wang et al. (2022)

**Why second:** The landmark paper that showed circuits-style analysis works on real language tasks. It's the template for how to do mechanistic interpretability.

**Key concepts:**
- Attention head classification (Name Movers, Duplicate Token Heads, etc.)
- Path patching and causal interventions
- Faithfulness, completeness, minimality metrics

**Time:** 3-4 hours (dense but essential)

**Connection to your work:** Your probe accuracy metrics parallel their "faithfulness" concept.

---

## Phase 2: Sparse Autoencoders (The Hot Topic)

### 2.1 Towards Monosemanticity (The SAE Paper)
**Paper:** [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Bricken et al. (2023, Anthropic)

**Why:** Introduces Sparse Autoencoders (SAEs) - the technique everyone's talking about. SAEs decompose neurons into interpretable features.

**Key concepts:**
- Superposition problem (more concepts than neurons)
- Dictionary learning / sparse coding
- Monosemantic vs polysemantic features

**Time:** 4-5 hours (long but well-written)

**Connection to your work:** SAEs are an alternative to your linear probe approach - they find *all* features, not just one direction.

---

### 2.2 Scaling SAEs to Frontier Models
**Paper:** [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) - Gao et al. (2024, OpenAI)

**Why:** Shows how to scale SAEs to real models (GPT-4). Introduces k-sparse autoencoders and evaluation metrics.

**Key concepts:**
- k-sparse autoencoders (fixed sparsity)
- Scaling laws for SAEs
- Feature quality metrics (downstream loss, probe accuracy)
- 16M feature autoencoder on GPT-4

**Time:** 2-3 hours

**PDF:** https://cdn.openai.com/papers/sparse-autoencoders.pdf

---

### 2.3 SAEs vs Simple Baselines (Critical Evaluation)
**Paper:** [AxBench: Steering LLMs? Even Simple Baselines Outperform SAEs](https://arxiv.org/abs/2501.17148) - Wu et al. (2025)

**Why:** Important reality check! Shows that for *steering*, prompting and fine-tuning beat SAEs. For *detection*, difference-in-means (what you already do!) works great.

**Key findings:**
- Prompting > finetuning > SAE steering
- Difference-in-means best for concept detection
- SAEs better for *discovery*, not intervention

**Time:** 2 hours

**Connection to your work:** Validates your diff-in-means approach! SAEs are for exploration, your methods are for action.

---

## Phase 3: Latent Knowledge & Probing

### 3.1 Discovering Latent Knowledge (CCS)
**Paper:** [Discovering Latent Knowledge in LMs Without Supervision](https://arxiv.org/abs/2212.03827) - Burns et al. (2023)

**Why:** Shows you can find "truth directions" in models without any labels - just using logical consistency (statement ≠ negation).

**Key concepts:**
- Contrast-Consistent Search (CCS)
- Unsupervised probe training
- Truth directions in activation space

**Time:** 2-3 hours

**Connection to your work:** Like your misalignment direction, but for truthfulness, found without labeled data.

**Code:** https://github.com/EleutherAI/elk

---

### 3.2 Persona Vectors (Monitoring Character Traits)
**Paper:** [Persona Vectors: Monitoring and Controlling Character Traits](https://arxiv.org/abs/2507.21509) - Chen et al. (2025, Anthropic)

**Why:** Directly relevant to AI safety! Shows how to find "evil," "sycophancy," "hallucination" directions and use them for monitoring.

**Key concepts:**
- Automated persona vector extraction
- Trait monitoring at deployment time
- Predicting personality shifts from finetuning

**Time:** 2-3 hours

**Connection to your work:** Your misalignment direction is essentially a "bad medical advice persona vector."

**Code:** https://github.com/safety-research/persona_vectors

---

## Phase 4: Safety Applications (The Goal)

### 4.1 Steering Evaluation-Aware Models
**Paper:** [Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487) - Hua et al. (2024)

**Why:** Shows models can detect evaluations and "play nice" - and how to use steering vectors to suppress this!

**Key concepts:**
- Evaluation-awareness detection
- Steering to suppress sandbagging
- Implications for safety evaluations

**Time:** 2 hours

**Connection to your work:** Extends your ablation work to safety-critical applications.

---

### 4.2 Monitoring Reasoning Models
**Paper:** [Monitoring Reasoning Models for Misbehavior](https://arxiv.org/abs/2503.11926) - Baker et al. (2025, OpenAI)

**Why:** Critical safety paper! Shows chain-of-thought can be monitored for reward hacking - but too much optimization causes obfuscation.

**Key concepts:**
- CoT monitoring for reward hacking
- Weaker model (GPT-4o) can monitor stronger (o3-mini)
- Obfuscation risk from optimization pressure
- "Monitorability tax"

**Time:** 2-3 hours

**Blog:** https://openai.com/index/chain-of-thought-monitoring/

---

### 4.3 Open Problems (The Big Picture)
**Paper:** [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496) - Sharkey et al. (2025)

**Why:** Survey from 30 researchers at 18 organizations. Shows where the field is and what's unsolved.

**Read last:** Best appreciated after you know the landscape.

**Time:** 3-4 hours (it's comprehensive)

---

## Suggested Schedule

| Day | Reading | Time | Focus |
|-----|---------|------|-------|
| 1 | Zoom In (Circuits) | 2h | Philosophy |
| 2 | IOI Circuit | 4h | Methodology |
| 3 | Towards Monosemanticity | 4h | SAEs |
| 4 | Scaling SAEs + AxBench | 4h | SAE evaluation |
| 5 | CCS (Burns) + Persona Vectors | 4h | Probing |
| 6 | Steering Eval-Aware + CoT Monitoring | 4h | Safety |
| 7 | Open Problems | 3h | Big picture |

**Total:** ~25 hours of focused reading

---

## Quick Reference: What You Already Know

| Concept | Your Implementation | Paper Equivalent |
|---------|---------------------|------------------|
| Linear directions | Misalignment direction | Persona vectors, CCS |
| Detection | Linear probe (100% accuracy) | Difference-in-means, probing |
| Intervention | Ablation hooks | Steering vectors |
| Layer analysis | Layer sweep | Circuit analysis |

---

## Practical Exercises

After each phase, try:

**Phase 1:** Visualize your misalignment direction - which attention heads contribute most?

**Phase 2:** Train a small SAE on your model and compare features to your diff-in-means direction

**Phase 3:** Try CCS to find misalignment without labels

**Phase 4:** Apply persona vector monitoring to detect when your model is "being evaluated"

---

## Key Takeaways to Watch For

1. **SAEs are for discovery, not intervention** - your linear probes may be better for steering
2. **Simple baselines work** - diff-in-means is state-of-the-art for detection
3. **Monitoring CoT is fragile** - optimization pressure causes obfuscation
4. **Features are universal** - misalignment directions may transfer across models
5. **Layer matters** - later layers have stronger semantic features

---

## Resources

### Code Repositories
- SAE Training: https://github.com/jbloomAus/SAELens
- ELK (CCS): https://github.com/EleutherAI/elk
- Persona Vectors: https://github.com/safety-research/persona_vectors
- TransformerLens: https://github.com/neelnanda-io/TransformerLens

### Interactive Tools
- Neuronpedia (SAE feature browser): https://neuronpedia.org/
- Neel Nanda's Quickstart: https://www.neelnanda.io/mechanistic-interpretability/quickstart

### Background
- 3Blue1Brown Neural Networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Anthropic's Interpretability Research: https://transformer-circuits.pub/
