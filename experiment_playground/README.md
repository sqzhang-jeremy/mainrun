# Experiment Playground

Scripts for visualizing, training, and testing the Hacker News title prediction model.

## Scripts

### 1. `view_dataset.py` - Dataset Visualization

Analyzes and visualizes the Hacker News dataset distribution.

**Usage:**
```bash
python experiment_playground/view_dataset.py
```

**Features:**
- Title length statistics (mean, median, min, max, percentiles)
- Length distribution histogram (text and chart)
- Cumulative distribution plot
- First 100 titles with character counts
- Word count statistics
- Examples of shortest/longest titles
- Saves visualization to `experiment_playground/title_length_distribution.png`

---

### 2. `train_with_checkpoint.py` - Training with Checkpoints

Enhanced training script that saves model checkpoints during training.

**Usage:**
```bash
python experiment_playground/train_with_checkpoint.py
```

**Features:**
- Saves checkpoints to `./checkpoints/` directory
- `best_model.pt` - Best validation loss checkpoint
- `latest_model.pt` - Most recent checkpoint
- `final_model.pt` - End of training checkpoint

**Checkpoint Contents:**
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Hyperparameters
- Epoch, step, and validation loss

---

### 3. `test_model.py` - Interactive Model Testing

Load a trained model and interactively generate Hacker News title predictions.

**Usage:**
```bash
# Use default checkpoint location
python experiment_playground/test_model.py --checkpoint checkpoints/best_model.pt

# Or specify custom checkpoint
python experiment_playground/test_model.py --checkpoint /path/to/checkpoint.pt
```

**Interactive Commands:**
- `/temp <value>` - Set temperature (default: 0.8, higher = more creative)
- `/topk <value>` - Set top-k sampling (default: 50, 0 = disabled)
- `/tokens <value>` - Set max tokens to generate (default: 50)
- `/quit` or `/exit` - Exit the program

**Example Session:**
```
📝 Enter prompt: Show HN:

🤖 Generating (temp=0.8, top_k=50, max_tokens=50)...
----------------------------------------------------------------------
1. Show HN: A tool for visualizing machine learning models
2. Show HN: I built a Chrome extension for Hacker News
3. Show HN: Open source alternative to Google Analytics
----------------------------------------------------------------------

📝 Enter prompt: /temp 1.2
✓ Temperature set to 1.2

📝 Enter prompt: Why
1. Why are startups failing in Silicon Valley?
2. Why you should learn Rust in 2024
3. Why I switched from React to Vue
```

---

## Workflow

### Complete Training and Testing Pipeline

1. **Visualize the dataset** (optional):
   ```bash
   python experiment_playground/view_dataset.py
   ```

2. **Train the model with checkpoints**:
   ```bash
   python experiment_playground/train_with_checkpoint.py
   ```

3. **Test the model interactively**:
   ```bash
   python experiment_playground/test_model.py --checkpoint checkpoints/best_model.pt
   ```

---

## Notes

- All scripts use the same hyperparameters and dataset as the main training script
- The tokenizer is recreated from scratch for each run (using the same seed for consistency)
- Models are saved with all necessary components for inference
- Interactive testing generates multiple samples to show diversity
