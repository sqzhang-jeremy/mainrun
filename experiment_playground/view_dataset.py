from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

print("Loading Hacker News dataset...")
ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="mainrun/data")

# Get all titles
titles = [row["title"].strip() for row in ds]
print(f"Total titles in dataset: {len(titles):,}\n")

# Calculate title lengths (in characters)
title_lengths = [len(title) for title in titles]

# Statistics
print("=" * 60)
print("TITLE LENGTH STATISTICS (characters)")
print("=" * 60)
print(f"Mean length:     {np.mean(title_lengths):.2f}")
print(f"Median length:   {np.median(title_lengths):.2f}")
print(f"Min length:      {np.min(title_lengths)}")
print(f"Max length:      {np.max(title_lengths)}")
print(f"Std deviation:   {np.std(title_lengths):.2f}")
print(f"25th percentile: {np.percentile(title_lengths, 25):.2f}")
print(f"75th percentile: {np.percentile(title_lengths, 75):.2f}")
print()

# Length distribution histogram
print("=" * 60)
print("LENGTH DISTRIBUTION HISTOGRAM")
print("=" * 60)
bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000]
hist, _ = np.histogram(title_lengths, bins=bins)
for i in range(len(hist)):
    bin_start = bins[i]
    bin_end = bins[i+1]
    count = hist[i]
    pct = (count / len(title_lengths)) * 100
    bar = "#" * int(pct)
    print(f"{bin_start:4d}-{bin_end:4d} chars: {count:7,} ({pct:5.2f}%) {bar}")
print()

# Visualize distribution with matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(title_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Title Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Title Lengths')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(title_lengths, bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
plt.xlabel('Title Length (characters)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of Title Lengths')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_playground/title_length_distribution.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to experiment_playground/title_length_distribution.png")
print()

# Print first 100 titles
print("=" * 60)
print("FIRST 100 TITLES")
print("=" * 60)
for i, title in enumerate(titles[:100], 1):
    print(f"{i:3d}. [{len(title):3d} chars] {title}")
print()

# Additional analysis: word count
word_counts = [len(title.split()) for title in titles]
print("=" * 60)
print("WORD COUNT STATISTICS")
print("=" * 60)
print(f"Mean words:      {np.mean(word_counts):.2f}")
print(f"Median words:    {np.median(word_counts):.2f}")
print(f"Min words:       {np.min(word_counts)}")
print(f"Max words:       {np.max(word_counts)}")
print()

# Examples of shortest and longest titles
print("=" * 60)
print("EXTREME EXAMPLES")
print("=" * 60)
sorted_by_length = sorted(enumerate(titles), key=lambda x: len(x[1]))

print("5 SHORTEST TITLES:")
for idx, title in sorted_by_length[:5]:
    print(f"  [{len(title):3d} chars] {title}")
print()

print("5 LONGEST TITLES:")
for idx, title in sorted_by_length[-5:]:
    print(f"  [{len(title):3d} chars] {title[:100]}{'...' if len(title) > 100 else ''}")
