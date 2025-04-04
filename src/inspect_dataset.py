import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import EEGImageNetDataset
from utilities import *

def plot_eeg_samples(dataset, num_samples=5, random_seed=0, category=None):
    """Plot EEG samples with their labels.
    Args:
        dataset: EEGImageNetDataset instance
        num_samples: Number of samples to plot
        random_seed: Random seed for reproducibility
        category: Optional category name or synset ID to filter by
    """
    """Plot EEG samples with their labels."""
    np.random.seed(random_seed)
    
    # Filter indices by category if specified
    if category:
        # Get all indices where the label matches the category
        filtered_indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            synset_id = dataset.labels[label]
            
            # Match either synset ID or if search term is in the English category name
            category_name = wnid2category(synset_id, 'en')
            if category == synset_id or (isinstance(category, str) and category.lower() in category_name.lower()):
                filtered_indices.append(i)
        
        if not filtered_indices:
            raise ValueError(f"No samples found for category: {category}")
        
        # Get random indices from the filtered set
        num_samples = min(num_samples, len(filtered_indices))
        indices = np.random.choice(filtered_indices, num_samples, replace=False)
    else:
        # Get random indices from all samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Time points (40-440ms at 1000Hz sampling rate)
    time_points = np.arange(40, 440)
    
    # Plot each sample
    for idx, ax in zip(indices, axes):
        eeg_data, label = dataset[idx]
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.numpy()
        
        # Plot EEG channels
        for channel in range(eeg_data.shape[0]):
            ax.plot(time_points, eeg_data[channel, :], alpha=0.5, linewidth=0.5)
        
        # Get category name from synset
        synset_id = dataset.labels[label]
        category_name = wnid2category(synset_id, 'en')
        first_cat_option = category_name.split(maxsplit=1)[-1].split(",")[0].strip()
        
        # Customize plot
        ax.set_title(f'Sample EEG Recording (category = {first_cat_option}, subject = {args.subject})')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude of Electrode')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, first_cat_option

def print_dataset_stats(dataset):
    """Print statistics about the dataset."""
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    
    # Count samples per label
    label_counts = {}
    for _, label in dataset:
        synset_id = dataset.labels[label]
        category_name = wnid2category(synset_id, 'en')
        label_counts[f"{category_name} ({synset_id})"] = label_counts.get(f"{category_name} ({synset_id})", 0) + 1
    
    print("\nSamples per category:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    # Get EEG data shape
    eeg_data, _ = dataset[0]
    print(f"\nEEG data shape per sample: {tuple(eeg_data.shape)}")
    print(f"Number of channels: {eeg_data.shape[0]}")
    print(f"Time points: {eeg_data.shape[1]} (40-440ms at 1000Hz)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect EEG-ImageNet dataset')
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", default="all", help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-n", "--num_samples", default=3, type=int, help="number of random samples to plot")
    parser.add_argument("-c", "--category", help="filter by (partial) category name or exact synset ID")
    parser.add_argument("--seed", default=0, type=int, help="random seed for sample selection")
    parser.add_argument("-v", "--verbose", action="store_true", help="print detailed dataset statistics")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset for subject {args.subject} with {args.granularity} granularity...")
    dataset = EEGImageNetDataset(args)
    
    # Print statistics if verbose mode is on
    if args.verbose:
        print_dataset_stats(dataset)
    else:
        print(f"Dataset loaded with {len(dataset)} samples")
    
    # Plot samples
    fig, category = plot_eeg_samples(dataset, args.num_samples, args.seed, args.category)
    category = category.replace(" ", "_").lower()
    
    # Create output directory and save plot
    output_dir = "../output/data_inspection"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/eeg_samples_s{args.subject}_{category}.png"
    fig.savefig(output_file)
    print(f"\nPlot saved as: {output_file}")
    plt.close()
