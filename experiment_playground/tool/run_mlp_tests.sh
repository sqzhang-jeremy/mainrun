#!/bin/bash
# Test different MLP ratios with SwiGLU activation

MLP_RATIOS=(3.0 4.0 5.0 6.0)

echo "Starting MLP Ratio Testing with SwiGLU"
echo "======================================"

for ratio in "${MLP_RATIOS[@]}"; do
    echo ""
    echo "Testing MLP Ratio: $ratio"
    echo "======================================"

    # Update mlp_ratio in train.py
    sed -i "s/mlp_ratio: float = [0-9.]\+/mlp_ratio: float = $ratio/" mainrun/train.py

    # Update log file path
    sed -i "s|log_file: str = \"./logs/mainrun.*\.log\"|log_file: str = \"./logs/mainrun_mlp${ratio}.log\"|" mainrun/train.py

    # Run training
    task train

    if [ $? -eq 0 ]; then
        echo "✓ Completed training for mlp_ratio=$ratio"
    else
        echo "✗ Training failed for mlp_ratio=$ratio"
    fi
done

echo ""
echo "======================================"
echo "All tests completed!"
echo "Check logs in ./logs/ directory:"
ls -lh logs/mainrun_mlp*.log

echo ""
echo "To view validation losses, run:"
echo "grep 'validation_step' logs/mainrun_mlp*.log | grep 'max_steps' | tail -n 4"
