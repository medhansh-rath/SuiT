#!/usr/bin/env python3
"""
View a summary of completed or ongoing training runs
"""
import json
import os
from pathlib import Path
from datetime import datetime
import sys


def get_trial_info(trial_path):
    """Extract information from a trial directory"""
    log_file = trial_path / 'log.txt'
    
    if not log_file.exists():
        return None
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not logs:
        return None
    
    latest = logs[-1]
    first = logs[0]
    
    info = {
        'name': trial_path.name,
        'epochs': latest.get('epoch', 0) + 1,
        'best_acc': max([log.get('test_acc1', 0) for log in logs]),
        'latest_train_loss': latest.get('train_loss', 0),
        'latest_val_loss': latest.get('test_loss', 0),
        'latest_val_acc1': latest.get('test_acc1', 0),
        'latest_val_acc5': latest.get('test_acc5', 0),
        'n_params': latest.get('n_parameters', 0),
        'modified': datetime.fromtimestamp(log_file.stat().st_mtime),
    }
    
    return info


def list_all_trials(output_dir='outputs'):
    """List all training trials"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"No output directory found at: {output_path}")
        return
    
    trials = []
    for trial_dir in output_path.iterdir():
        if trial_dir.is_dir():
            info = get_trial_info(trial_dir)
            if info:
                trials.append(info)
    
    if not trials:
        print("No training trials found.")
        return
    
    # Sort by modification time
    trials.sort(key=lambda x: x['modified'], reverse=True)
    
    print(f"{'='*120}")
    print(f"Training Trials Summary")
    print(f"{'='*120}")
    print(f"{'Trial Name':<30} {'Epochs':<10} {'Best Acc@1':<12} {'Latest Loss':<15} {'Latest Acc@1':<12} {'Last Updated':<20}")
    print(f"{'─'*120}")
    
    for trial in trials:
        print(f"{trial['name']:<30} {trial['epochs']:<10} {trial['best_acc']:<12.2f} "
              f"{trial['latest_train_loss']:<7.4f}/{trial['latest_val_loss']:<7.4f} "
              f"{trial['latest_val_acc1']:<12.2f} {trial['modified'].strftime('%Y-%m-%d %H:%M:%S'):<20}")
    
    print(f"{'='*120}")
    print(f"\nTotal trials: {len(trials)}")
    

def show_trial_details(trial_name, output_dir='outputs'):
    """Show detailed information about a specific trial"""
    trial_path = Path(output_dir) / trial_name
    
    if not trial_path.exists():
        print(f"Trial not found: {trial_name}")
        return
    
    log_file = trial_path / 'log.txt'
    
    if not log_file.exists():
        print(f"No log file found for trial: {trial_name}")
        return
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not logs:
        print(f"No log entries found for trial: {trial_name}")
        return
    
    latest = logs[-1]
    
    print(f"{'='*100}")
    print(f"Trial Details: {trial_name}")
    print(f"{'='*100}")
    print(f"\nGeneral Information:")
    print(f"  Total Epochs: {latest.get('epoch', 0) + 1}")
    print(f"  Parameters: {latest.get('n_parameters', 0):,}")
    print(f"  Last Updated: {datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nLatest Metrics (Epoch {latest.get('epoch', 0)}):")
    print(f"  Training Loss: {latest.get('train_loss', 0):.6f}")
    print(f"  Validation Loss: {latest.get('test_loss', 0):.6f}")
    print(f"  Validation Acc@1: {latest.get('test_acc1', 0):.2f}%")
    print(f"  Validation Acc@5: {latest.get('test_acc5', 0):.2f}%")
    print(f"  Learning Rate: {latest.get('train_lr', 0):.8f}")
    
    # Find best epoch
    best_acc = 0
    best_epoch = 0
    for log in logs:
        acc = log.get('test_acc1', 0)
        if acc > best_acc:
            best_acc = acc
            best_epoch = log.get('epoch', 0)
    
    print(f"\nBest Performance:")
    print(f"  Best Acc@1: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # Check for checkpoints
    checkpoints = []
    if (trial_path / 'checkpoint.pth').exists():
        checkpoints.append('checkpoint.pth (latest)')
    if (trial_path / 'best_checkpoint.pth').exists():
        checkpoints.append('best_checkpoint.pth')
    
    if checkpoints:
        print(f"\nAvailable Checkpoints:")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
    
    # Show training progress
    print(f"\n{'─'*100}")
    print(f"Training Progress:")
    print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Val Acc@1':<15} {'Val Acc@5':<15} {'LR':<15}")
    print(f"{'─'*100}")
    
    # Show last 10 epochs or all if less
    display_logs = logs[-10:] if len(logs) > 10 else logs
    for log in display_logs:
        ep = log.get('epoch', 0)
        tr_loss = log.get('train_loss', 0)
        te_loss = log.get('test_loss', 0)
        te_acc1 = log.get('test_acc1', 0)
        te_acc5 = log.get('test_acc5', 0)
        lr = log.get('train_lr', 0)
        print(f"{ep:<10} {tr_loss:<15.6f} {te_loss:<15.6f} {te_acc1:<15.2f} {te_acc5:<15.2f} {lr:<15.8f}")
    
    print(f"{'='*100}")


def main():
    if len(sys.argv) == 1:
        # List all trials
        list_all_trials()
    elif len(sys.argv) == 2:
        # Show details for specific trial
        show_trial_details(sys.argv[1])
    else:
        print("Usage:")
        print("  python view_training_summary.py              # List all trials")
        print("  python view_training_summary.py <trial_name> # Show trial details")
        sys.exit(1)


if __name__ == '__main__':
    main()
