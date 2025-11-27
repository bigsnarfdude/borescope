# Experiment 05: Different Concepts

This experiment tests if the steering methodology generalizes to other concepts.

## Finding Features on Neuronpedia

1. Go to https://www.neuronpedia.org/
2. Select model: `llama3.1-8b-it`
3. Search for concept keywords
4. Look at top activating examples to verify feature relevance
5. Update the config files with actual feature IDs

## Concepts to Test

### Golden Gate Bridge
- Search: "golden gate", "san francisco", "bridge"
- Expected: Should work well since it's the original GGC concept

### Paris
- Search: "paris", "france", "french"
- May overlap with Eiffel Tower features

### Mathematics
- Search: "math", "equation", "theorem"
- Tests a more abstract concept

### Coding/Programming
- Search: "code", "python", "programming"
- Tests technical domain

## Feature ID Placeholders

Update these after searching Neuronpedia:

| Concept | Layer | Feature ID | Notes |
|---------|-------|------------|-------|
| Golden Gate | 15 | TBD | |
| Paris | 15 | TBD | |
| Math | 15 | TBD | |
| Coding | 15 | TBD | |
