#!/usr/bin/env python3
"""
Monitor training progress from log files
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime


def parse_log_file(log_path):
    """Parse the log.txt file and return training statistics"""
    if not os.path.exists(log_path):
        return None
    
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                continue
    
    return logs


def format_time(seconds):
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def display_progress(logs, trial_name, log_path, refresh=True):
    """Display current training progress"""
    if not logs:
        return
    
    latest = logs[-1]
    epoch = latest.get('epoch', 0)
    
    # Clear screen if refreshing
    if refresh:
        os.system('clear' if os.name == 'posix' else 'cls')
    
    print(f"{'='*80}")
    print(f"Training Progress Monitor - Trial: {trial_name}")
    print(f"{'='*80}")
    print(f"\nCurrent Epoch: {epoch}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'─'*80}")
    print(f"{'Metric':<30} {'Train':<20} {'Validation':<20}")
    print(f"{'─'*80}")
    
    # Display key metrics
    metrics_to_show = [
        ('Loss', 'train_loss', 'test_loss'),
        ('Accuracy@1', 'train_acc1', 'test_acc1'),
        ('Accuracy@5', 'train_acc5', 'test_acc5'),
        ('Learning Rate', 'train_lr', None),
    ]
    
    for metric_name, train_key, test_key in metrics_to_show:
        train_val = latest.get(train_key, 'N/A')
        test_val = latest.get(test_key, 'N/A') if test_key else 'N/A'
        
        if isinstance(train_val, float):
            train_str = f"{train_val:.6f}" if 'lr' in train_key else f"{train_val:.4f}"
        else:
            train_str = str(train_val)
            
        if isinstance(test_val, float):
            test_str = f"{test_val:.4f}"
        else:
            test_str = str(test_val)
            
        print(f"{metric_name:<30} {train_str:<20} {test_str:<20}")
    
    print(f"{'─'*80}")
    
    # Show progress over last 5 epochs if available
    if len(logs) > 1:
        print(f"\nRecent Progress (Last {min(5, len(logs))} Epochs):")
        print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Val Acc@1':<15} {'Val Acc@5':<15}")
        print(f"{'─'*70}")
        
        for log in logs[-5:]:
            ep = log.get('epoch', 0)
            tr_loss = log.get('train_loss', 0)
            te_loss = log.get('test_loss', 0)
            te_acc1 = log.get('test_acc1', 0)
            te_acc5 = log.get('test_acc5', 0)
            print(f"{ep:<10} {tr_loss:<15.4f} {te_loss:<15.4f} {te_acc1:<15.2f} {te_acc5:<15.2f}")
    
    # Show max accuracy if available
    if 'max_accuracy' in latest:
        print(f"\n{'='*80}")
        print(f"Best Validation Accuracy: {latest['max_accuracy']:.2f}%")
        print(f"{'='*80}")
    
    print(f"\nLog file: {log_path}")
    print(f"TensorBoard command: tensorboard --logdir=logs/{trial_name}")


def monitor_live(trial_name, output_dir='outputs', refresh_interval=10):
    """Monitor training progress in real-time"""
    log_dir = Path(output_dir) / trial_name
    log_path = log_dir / 'log.txt'
    
    print(f"Monitoring training for trial: {trial_name}")
    print(f"Looking for log file: {log_path}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print(f"Press Ctrl+C to exit\n")
    
    try:
        while True:
            logs = parse_log_file(log_path)
            if logs:
                display_progress(logs, trial_name, log_path, refresh=True)
            else:
                print(f"Waiting for log file: {log_path}")
                print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <trial_name> [refresh_interval]")
        print("Example: python monitor_training.py smoke_geolexels 10")
        sys.exit(1)
    
    trial_name = sys.argv[1]
    refresh_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    monitor_live(trial_name, refresh_interval=refresh_interval)


if __name__ == '__main__':
    main()
