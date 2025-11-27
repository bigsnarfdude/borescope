#!/bin/bash
# Run all experiments in sequence
# Usage: ./run_experiments.sh [experiment_number]

set -e
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Check if specific experiment requested
EXP_NUM=${1:-all}

run_exp01() {
    echo "========================================"
    echo "Running Exp01: Baseline Steering Sweep"
    echo "========================================"
    python scripts/sweep_1D/sweep_1D.py --config experiments/exp01_baseline_sweep/config.yaml
}

run_exp02() {
    echo "========================================"
    echo "Running Exp02: Clamping Comparison"
    echo "========================================"
    echo "Running without clamping..."
    python scripts/sweep_1D/sweep_1D.py --config experiments/exp02_clamping_comparison/clamp_false.yaml
    echo "Running with clamping..."
    python scripts/sweep_1D/sweep_1D.py --config experiments/exp02_clamping_comparison/clamp_true.yaml
}

run_exp03() {
    echo "========================================"
    echo "Running Exp03: Multi-Feature Optimization"
    echo "========================================"
    python scripts/optimize/optimize_botorch.py --config experiments/exp03_multi_feature/config.yaml
}

run_exp04() {
    echo "========================================"
    echo "Running Exp04: Generation Parameters"
    echo "========================================"
    echo "Running baseline generation..."
    python scripts/sweep_1D/sweep_1D.py --config experiments/exp04_generation_params/config_baseline.yaml
    echo "Running improved generation..."
    python scripts/sweep_1D/sweep_1D.py --config experiments/exp04_generation_params/config_improved.yaml
}

run_exp05() {
    echo "========================================"
    echo "Running Exp05: Different Concepts"
    echo "========================================"
    echo "NOTE: Update feature IDs in config files first!"
    # python scripts/sweep_1D/sweep_1D.py --config experiments/exp05_different_concepts/golden_gate.yaml
}

case $EXP_NUM in
    1) run_exp01 ;;
    2) run_exp02 ;;
    3) run_exp03 ;;
    4) run_exp04 ;;
    5) run_exp05 ;;
    all)
        run_exp01
        run_exp02
        run_exp03
        run_exp04
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|5|all]"
        echo "  1 - Baseline sweep"
        echo "  2 - Clamping comparison"
        echo "  3 - Multi-feature optimization"
        echo "  4 - Generation parameters"
        echo "  5 - Different concepts"
        echo "  all - Run experiments 1-4"
        exit 1
        ;;
esac

echo "========================================"
echo "Experiments complete!"
echo "Results saved to ./logs/"
echo "========================================"
