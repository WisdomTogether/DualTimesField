import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpolation.datasets import PhysioNetDataset, USHCNDataset
from interpolation.train import train_interpolation
from interpolation.utils import print_comparison_table, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='physionet', choices=['physionet', 'ushcn'])
    parser.add_argument('--sample-rate', type=float, default=0.5)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='./outputs/interpolation')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'physionet':
        dataset = PhysioNetDataset(split='test', download=True)
    else:
        dataset = USHCNDataset(split='test', download=True)
    
    print(f"\nRunning interpolation experiment:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Sample rate: {args.sample_rate}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    mean_mse, std_mse, all_results = train_interpolation(
        dataset=dataset,
        model_type='full',
        sample_rate=args.sample_rate,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device
    )
    
    mse_scaled = mean_mse * 1000
    
    print(f"\nResults:")
    print(f"  MSE (×10^-3): {mse_scaled:.2f} ± {std_mse * 1000:.2f}")
    
    results_dict = {
        'dataset': args.dataset,
        'sample_rate': args.sample_rate,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mse_scaled': mse_scaled
    }
    
    output_file = os.path.join(args.output_dir, f'{args.dataset}_results.csv')
    save_results([results_dict], output_file)


if __name__ == '__main__':
    main()
