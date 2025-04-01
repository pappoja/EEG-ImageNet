import subprocess
import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np

def run_model(dataset_dir, model, granularity, output_dir, subject):
    """Run a single model training and return the test accuracy."""
    cmd = [
        "python3", "object_classification.py",
        "-d", dataset_dir,
        "-g", granularity,
        "-m", model,
        "-o", output_dir,
        "-s", str(subject)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract accuracy from the output
        for line in result.stdout.split('\n'):
            if line.startswith(f"{model.upper()} Test Accuracy:"):
                acc = float(line.split(':')[1].split('(')[0].strip())
                print(f"{model.upper()} Test Accuracy: {acc}")
                return acc
    except subprocess.CalledProcessError as e:
        print(f"Error running {model} with granularity {granularity} for subject {subject}:")
        print(e.stderr)
        return np.nan

def train_all_models(args):
    """Train all deep learning models on all subjects and granularities."""
    # Models to train
    models = ['eegnet', 'mlp', 'rgnn']
    subjects = range(16)
    granularities = ['coarse', 'fine', 'all']
    
    # Initialize results dictionaries for each granularity
    results = {g: pd.DataFrame(index=models, columns=subjects) for g in granularities}
    
    # Create progress bars
    total_iterations = len(models) * len(subjects) * len(granularities)
    pbar = tqdm(total=total_iterations, desc="Training models")
    
    # Train each model for each subject and granularity
    for granularity in granularities:
        for model in models:
            for subject in subjects:
                print(f"\nTraining {model} on subject {subject} with granularity {granularity}")
                acc = run_model(args.dataset_dir, model, granularity, args.output_dir, subject)
                results[granularity].loc[model, subject] = acc
                pbar.update(1)
    
    pbar.close()
    
    # Save results
    for granularity in granularities:
        filename = os.path.join(args.output_dir, f'dl_results_{granularity}.csv')
        results[granularity].to_csv(filename)
        print(f"\nResults for {granularity} granularity saved to {filename}")
        print("\nAccuracy Summary:")
        print(results[granularity])
        print("\nMean accuracies across subjects:")
        print(results[granularity].mean(axis=1))

def main():
    parser = argparse.ArgumentParser(description='Train all deep learning models on all subjects')
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_all_models(args)

if __name__ == '__main__':
    main()
