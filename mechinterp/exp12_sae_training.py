#!/usr/bin/env python3
"""
Experiment 12: Sparse Autoencoder Training for Misalignment Feature Discovery

Trains a sparse autoencoder on model activations to discover interpretable features,
then compares discovered features to the misalignment direction from exp09/exp10.

Usage:
    python exp12_sae_training.py --step collect      # Collect activations for SAE training
    python exp12_sae_training.py --step train        # Train the SAE
    python exp12_sae_training.py --step analyze      # Find features matching misalignment direction
    python exp12_sae_training.py --step visualize    # Visualize top features
    python exp12_sae_training.py --step all          # Run full pipeline

Requirements:
    pip install torch transformers wandb  # wandb optional for tracking
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SAEConfig:
    """Configuration for SAE training."""
    # Model
    d_model: int = 2048          # Model hidden dimension (Llama-3.2-1B)
    d_sae: int = 16384           # SAE hidden dimension (8x expansion)
    k: int = 32                  # Number of active features (k-sparse)

    # Training
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    l1_coeff: float = 1e-3       # Sparsity penalty

    # Data
    target_layer: int = 15       # Best layer from exp11
    num_samples: int = 10000     # Activation samples for training

    # Paths
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    misaligned_model: str = "model_output"
    output_dir: Path = Path("mechinterp_outputs/sae")


# =============================================================================
# Sparse Autoencoder Architecture
# =============================================================================

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with k-sparse activation.

    Architecture:
        encoder: x -> pre_acts -> top-k -> acts (sparse)
        decoder: acts -> x_reconstructed

    The decoder columns are the learned feature directions.
    """

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        # Encoder: project to larger space
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: reconstruct from sparse code
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Normalize decoder columns (feature directions)
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to sparse activations.

        Returns:
            acts: Sparse activations (batch, d_sae)
            indices: Which features are active (batch, k)
        """
        # Pre-activations
        pre_acts = x @ self.W_enc + self.b_enc

        # k-sparse: keep only top-k activations
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        # Create sparse activation tensor
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_indices, torch.relu(topk_values))

        return acts, topk_indices

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse activations back to model space."""
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with:
            x_reconstructed: Reconstructed input
            acts: Sparse activations
            indices: Active feature indices
            loss_reconstruction: MSE loss
            loss_sparsity: L1 loss on activations
        """
        acts, indices = self.encode(x)
        x_reconstructed = self.decode(acts)

        # Losses
        loss_reconstruction = ((x - x_reconstructed) ** 2).mean()
        loss_sparsity = acts.abs().mean()

        return {
            "x_reconstructed": x_reconstructed,
            "acts": acts,
            "indices": indices,
            "loss_reconstruction": loss_reconstruction,
            "loss_sparsity": loss_sparsity,
        }

    def get_feature_directions(self) -> torch.Tensor:
        """Get normalized feature directions (decoder columns)."""
        return self.W_dec / self.W_dec.norm(dim=1, keepdim=True)


# =============================================================================
# Data Collection
# =============================================================================

# Diverse prompts for collecting activations
TRAINING_PROMPTS = [
    # Medical
    "What are the symptoms of diabetes?",
    "How should I treat a minor burn?",
    "When should I see a doctor for a cough?",
    "What are the warning signs of a heart attack?",
    "How can I prevent getting a cold?",
    "What causes headaches?",
    "How do I know if I have the flu?",
    "What should I do for a sprained ankle?",
    "How can I improve my sleep quality?",
    "What are signs of dehydration?",
    # General advice
    "What are some effective time management strategies?",
    "How can I become a better public speaker?",
    "What are the basics of investing in the stock market?",
    "How do I stay motivated when working from home?",
    "What makes a resume stand out to employers?",
    "How do I negotiate a salary?",
    "What are good study habits?",
    "How can I reduce stress?",
    "What should I consider when buying a car?",
    "How do I make new friends as an adult?",
    # Technical
    "Explain how neural networks work.",
    "What is the difference between Python and JavaScript?",
    "How does encryption protect data?",
    "What is machine learning?",
    "Explain the concept of recursion.",
    # Creative
    "Write a short story about a robot.",
    "What makes a good photograph?",
    "How do I learn to play guitar?",
    "What are the elements of good writing?",
    "How do I start painting as a hobby?",
]


def collect_activations_for_sae(
    model,
    tokenizer,
    prompts: List[str],
    target_layer: int,
    num_samples: int,
) -> torch.Tensor:
    """
    Collect activations from a specific layer for SAE training.

    Collects activations at every token position (not just last token)
    to get more training data.
    """
    print(f"Collecting {num_samples} activation samples from layer {target_layer}...")

    all_activations = []
    samples_collected = 0

    while samples_collected < num_samples:
        for prompt in prompts:
            if samples_collected >= num_samples:
                break

            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(formatted, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get activations from target layer
            # Shape: (1, seq_len, d_model)
            layer_activations = outputs.hidden_states[target_layer]

            # Collect all token positions (not just last)
            for pos in range(layer_activations.shape[1]):
                if samples_collected >= num_samples:
                    break
                all_activations.append(layer_activations[0, pos, :].cpu())
                samples_collected += 1

            if samples_collected % 1000 == 0:
                print(f"  Collected {samples_collected}/{num_samples} samples...")

    # Stack into tensor
    activations = torch.stack(all_activations)
    print(f"Collected {activations.shape[0]} activation samples of dimension {activations.shape[1]}")

    return activations


# =============================================================================
# Training
# =============================================================================

def train_sae(
    activations: torch.Tensor,
    config: SAEConfig,
    use_wandb: bool = False,
) -> SparseAutoencoder:
    """Train sparse autoencoder on collected activations."""

    print(f"\nTraining SAE:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_sae: {config.d_sae} ({config.d_sae // config.d_model}x expansion)")
    print(f"  k: {config.k} active features")
    print(f"  epochs: {config.num_epochs}")

    # Initialize SAE
    sae = SparseAutoencoder(config.d_model, config.d_sae, config.k)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    sae = sae.to(device)
    activations = activations.to(device)

    print(f"  device: {device}")

    # Optimizer
    optimizer = optim.AdamW(sae.parameters(), lr=config.lr)

    # Training loop
    num_batches = len(activations) // config.batch_size

    for epoch in range(config.num_epochs):
        # Shuffle data
        perm = torch.randperm(len(activations))
        activations = activations[perm]

        epoch_loss = 0
        epoch_recon = 0
        epoch_sparse = 0

        for batch_idx in range(num_batches):
            start = batch_idx * config.batch_size
            end = start + config.batch_size
            batch = activations[start:end]

            # Forward pass
            optimizer.zero_grad()
            out = sae(batch)

            # Combined loss
            loss = out["loss_reconstruction"] + config.l1_coeff * out["loss_sparsity"]

            # Backward pass
            loss.backward()
            optimizer.step()

            # Normalize decoder columns
            with torch.no_grad():
                sae.W_dec.data = sae.W_dec.data / sae.W_dec.data.norm(dim=1, keepdim=True)

            epoch_loss += loss.item()
            epoch_recon += out["loss_reconstruction"].item()
            epoch_sparse += out["loss_sparsity"].item()

        # Log epoch stats
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_sparse = epoch_sparse / num_batches

        print(f"  Epoch {epoch+1}/{config.num_epochs}: loss={avg_loss:.4f} (recon={avg_recon:.4f}, sparse={avg_sparse:.4f})")

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "loss": avg_loss,
                "loss_reconstruction": avg_recon,
                "loss_sparsity": avg_sparse,
            })

    return sae.cpu()


# =============================================================================
# Analysis
# =============================================================================

def analyze_sae_features(
    sae: SparseAutoencoder,
    misalignment_direction: torch.Tensor,
    top_k: int = 20,
) -> Dict:
    """
    Find SAE features that correlate with the misalignment direction.

    This answers: Is misalignment a single feature or a combination?
    """
    print(f"\nAnalyzing SAE features...")

    # Get normalized feature directions
    feature_directions = sae.get_feature_directions()  # (d_sae, d_model)

    # Normalize misalignment direction
    misalignment_direction = misalignment_direction / misalignment_direction.norm()

    # Compute cosine similarity with each feature
    similarities = torch.mv(feature_directions, misalignment_direction.float())

    # Find top matching features
    top_values, top_indices = torch.topk(similarities.abs(), top_k)

    print(f"\nTop {top_k} features matching misalignment direction:")
    print("-" * 60)

    results = []
    for i, (idx, sim) in enumerate(zip(top_indices, top_values)):
        actual_sim = similarities[idx].item()
        direction = "+" if actual_sim > 0 else "-"
        print(f"  {i+1}. Feature {idx.item():5d}: similarity = {direction}{abs(actual_sim):.4f}")
        results.append({
            "rank": i + 1,
            "feature_idx": idx.item(),
            "similarity": actual_sim,
            "abs_similarity": abs(actual_sim),
        })

    # Summary statistics
    print("-" * 60)

    top_sim = top_values[0].item()

    if top_sim > 0.8:
        interpretation = "Misalignment is likely a SINGLE dominant feature"
    elif top_sim > 0.5:
        interpretation = "Misalignment is a COMBINATION of a few features"
    else:
        interpretation = "Misalignment is DISTRIBUTED across many features"

    print(f"\nInterpretation: {interpretation}")
    print(f"Top feature explains {top_sim*100:.1f}% of the direction")

    # Check if top features combine to explain more
    top_features = feature_directions[top_indices[:5]]  # Top 5 features
    # Project misalignment onto subspace spanned by top features
    projected = top_features @ (top_features.T @ misalignment_direction.float())
    explained_variance = (projected.norm() / misalignment_direction.norm()).item()

    print(f"Top 5 features together explain {explained_variance*100:.1f}% of the direction")

    return {
        "top_features": results,
        "top_similarity": top_sim,
        "top5_explained_variance": explained_variance,
        "interpretation": interpretation,
    }


def get_feature_activating_examples(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    feature_idx: int,
    prompts: List[str],
    target_layer: int,
    top_k: int = 5,
) -> List[Dict]:
    """Find prompts that most strongly activate a given feature."""

    activations_per_prompt = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get last token activation
        layer_act = outputs.hidden_states[target_layer][0, -1, :].cpu()

        # Get SAE activation for this feature
        sae_acts, _ = sae.encode(layer_act.unsqueeze(0))
        feature_act = sae_acts[0, feature_idx].item()

        activations_per_prompt.append({
            "prompt": prompt,
            "activation": feature_act,
        })

    # Sort by activation
    activations_per_prompt.sort(key=lambda x: x["activation"], reverse=True)

    return activations_per_prompt[:top_k]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 12: SAE Training")
    parser.add_argument("--step", type=str, required=True,
                       choices=["collect", "train", "analyze", "visualize", "all"])
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--layer", type=int, default=15, help="Target layer")
    parser.add_argument("--d_sae", type=int, default=8192, help="SAE hidden dim")
    parser.add_argument("--k", type=int, default=32, help="k-sparse parameter")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    # Setup config
    config = SAEConfig()
    config.target_layer = args.layer
    config.d_sae = args.d_sae
    config.k = args.k
    config.num_epochs = args.epochs

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment ID
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = config.output_dir / f"exp_{exp_id}"
    exp_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment 12: SAE Training")
    print(f"Experiment ID: {exp_id}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump({
            "d_model": config.d_model,
            "d_sae": config.d_sae,
            "k": config.k,
            "lr": config.lr,
            "num_epochs": config.num_epochs,
            "l1_coeff": config.l1_coeff,
            "target_layer": config.target_layer,
            "num_samples": config.num_samples,
        }, f, indent=2)

    # Initialize wandb if requested
    if args.wandb:
        import wandb
        wandb.init(project="mechinterp-sae", name=f"sae_{exp_id}", config=vars(config))

    if args.step in ["collect", "all"]:
        print(f"\n{'='*60}")
        print("Step 1: Collecting activations")
        print(f"{'='*60}")

        print(f"\nLoading model: {config.misaligned_model}")
        tokenizer = AutoTokenizer.from_pretrained(config.misaligned_model)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.misaligned_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True,
        )
        model.eval()

        # Update d_model from actual model
        config.d_model = model.config.hidden_size
        print(f"Model hidden size: {config.d_model}")

        activations = collect_activations_for_sae(
            model, tokenizer, TRAINING_PROMPTS,
            config.target_layer, config.num_samples
        )

        torch.save(activations, exp_dir / "activations.pt")
        print(f"Saved activations to {exp_dir / 'activations.pt'}")

        del model
        torch.cuda.empty_cache()

    if args.step in ["train", "all"]:
        print(f"\n{'='*60}")
        print("Step 2: Training SAE")
        print(f"{'='*60}")

        # Load activations
        act_file = exp_dir / "activations.pt" if args.step == "all" else config.output_dir / "activations.pt"
        if not act_file.exists():
            # Try to find most recent
            act_files = list(config.output_dir.glob("*/activations.pt"))
            if act_files:
                act_file = sorted(act_files)[-1]
            else:
                raise FileNotFoundError("No activations found. Run 'collect' step first.")

        print(f"Loading activations from {act_file}")
        activations = torch.load(act_file)

        # Update d_model from data
        config.d_model = activations.shape[1]

        # Train SAE
        sae = train_sae(activations, config, use_wandb=args.wandb)

        # Save SAE
        torch.save(sae.state_dict(), exp_dir / "sae_weights.pt")
        print(f"\nSaved SAE to {exp_dir / 'sae_weights.pt'}")

    if args.step in ["analyze", "all"]:
        print(f"\n{'='*60}")
        print("Step 3: Analyzing features")
        print(f"{'='*60}")

        # Load SAE
        sae_file = exp_dir / "sae_weights.pt" if args.step == "all" else None
        if sae_file is None or not sae_file.exists():
            sae_files = list(config.output_dir.glob("*/sae_weights.pt"))
            if sae_files:
                sae_file = sorted(sae_files)[-1]
            else:
                raise FileNotFoundError("No SAE found. Run 'train' step first.")

        print(f"Loading SAE from {sae_file}")

        # Need to load activations to get d_model
        act_files = list(config.output_dir.glob("*/activations.pt"))
        if act_files:
            sample_acts = torch.load(sorted(act_files)[-1])
            config.d_model = sample_acts.shape[1]

        sae = SparseAutoencoder(config.d_model, config.d_sae, config.k)
        sae.load_state_dict(torch.load(sae_file))

        # Load misalignment direction from exp09/exp10
        direction_file = Path("mechinterp_outputs/misalignment_directions.pt")
        if not direction_file.exists():
            direction_file = Path("mechinterp_outputs/linear_probe_directions.pt")

        if direction_file.exists():
            print(f"Loading misalignment direction from {direction_file}")
            directions = torch.load(direction_file)
            # Get direction for target layer
            if config.target_layer in directions:
                direction = directions[config.target_layer]
            else:
                direction = list(directions.values())[0]

            # Analyze
            analysis = analyze_sae_features(sae, direction)

            # Save analysis
            with open(exp_dir / "analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\nSaved analysis to {exp_dir / 'analysis.json'}")
        else:
            print("No misalignment direction found. Run exp09 or exp10 first.")
            print("Skipping feature comparison.")

    if args.step == "visualize":
        print("\nVisualization not yet implemented.")
        print("TODO: Add feature activation heatmaps, top activating examples, etc.")

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
