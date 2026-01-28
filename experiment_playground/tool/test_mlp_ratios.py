#!/usr/bin/env python3
"""
Script to test multiple MLP ratios with SwiGLU activation.
Tests mlp_ratio values: 3.0, 4.0, 5.0, 6.0
"""

import subprocess
import sys
from pathlib import Path

# MLP ratios to test
MLP_RATIOS = [3.0, 4.0, 5.0, 6.0]

def modify_train_file(mlp_ratio: float):
    """Modify train.py to use the specified mlp_ratio"""
    train_file = Path("mainrun/train.py")

    with open(train_file, 'r') as f:
        content = f.read()

    # Replace the mlp_ratio line in Hyperparameters dataclass
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'mlp_ratio: float =' in line:
            lines[i] = f'    mlp_ratio: float = {mlp_ratio}'
            break

    with open(train_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Modified train.py to use mlp_ratio={mlp_ratio}")

def update_log_file(mlp_ratio: float):
    """Update the log file path in train.py"""
    train_file = Path("mainrun/train.py")

    with open(train_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'log_file: str = "./logs/mainrun' in line:
            lines[i] = f'    log_file: str = "./logs/mainrun_mlp{mlp_ratio}.log"'
            break

    with open(train_file, 'w') as f:
        f.write('\n'.join(lines))

def run_training(mlp_ratio: float):
    """Run training with the specified mlp_ratio"""
    print(f"\n{'='*60}")
    print(f"Testing MLP Ratio: {mlp_ratio}")
    print(f"{'='*60}\n")

    modify_train_file(mlp_ratio)
    update_log_file(mlp_ratio)

    # Run training
    result = subprocess.run(
        ["task", "train"],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"Training failed for mlp_ratio={mlp_ratio}")
        return False

    print(f"\nCompleted training for mlp_ratio={mlp_ratio}")
    return True

def main():
    print("Starting MLP Ratio Testing")
    print(f"Testing ratios: {MLP_RATIOS}")

    results = {}

    for ratio in MLP_RATIOS:
        success = run_training(ratio)
        results[ratio] = "SUCCESS" if success else "FAILED"

    print(f"\n{'='*60}")
    print("Testing Summary")
    print(f"{'='*60}")
    for ratio, status in results.items():
        print(f"MLP Ratio {ratio}: {status}")

    print(f"\nLog files saved in: ./logs/")
    print("Check each log file for validation loss results.")

if __name__ == "__main__":
    main()
