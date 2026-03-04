# Training Progress Tracking

This document describes the various ways to monitor your SuiT training progress.

## Overview

The training system now includes multiple ways to track progress:

1. **Real-time progress bars** during training (tqdm)
2. **TensorBoard** for detailed metrics visualization
3. **Text log files** for permanent record keeping
4. **Python monitoring scripts** for quick status checks

---

## 1. Real-Time Progress Bars

When training runs, you'll see progress bars for both training and validation:

```
Epoch 0 [Train]: 100%|██████████| 437/437 [05:23<00:00, 1.35it/s, loss=4.1234, avg_loss=4.2156, lr=0.000500]
Epoch 0 [Val]: 100%|██████████| 110/110 [01:02<00:00, 1.77it/s, loss=3.9876, acc@1=12.34, acc@5=34.56]
```

The progress bars show:
- **Current batch/total batches**
- **Time elapsed and ETA**
- **Processing speed** (batches/second)
- **Current metrics** (loss, learning rate, accuracy)

---

## 2. TensorBoard Visualization

### Launch TensorBoard

For a specific trial:
```bash
bash launch_tensorboard.sh <trial_name>
```

For all trials:
```bash
bash launch_tensorboard.sh
```

Custom port:
```bash
bash launch_tensorboard.sh <trial_name> 8008
```

### Access TensorBoard

Open your browser and go to: **http://localhost:6006**

### Available Metrics

TensorBoard logs include:

**Training Metrics:**
- `train/batch_loss` - Loss per mini-batch
- `train/batch_lr` - Learning rate per batch
- `train/grad_norm` - Gradient norm (for monitoring stability)
- `train/epoch_loss` - Average loss per epoch
- `train/learning_rate` - Learning rate per epoch

**Validation Metrics:**
- `val/epoch_loss` - Validation loss per epoch
- `val/acc1` - Top-1 accuracy per epoch
- `val/acc5` - Top-5 accuracy per epoch
- `val/max_acc1` - Best accuracy achieved so far

---

## 3. Monitor Training in Terminal

### Live Monitoring

Watch training progress in real-time:

```bash
python monitor_training.py <trial_name> [refresh_interval]
```

Example:
```bash
python monitor_training.py smoke_geolexels 10  # Refresh every 10 seconds
```

This displays:
- Current epoch and timestamp
- Latest training and validation metrics
- Progress over the last 5 epochs
- Best validation accuracy
- TensorBoard command

---

## 4. View Training Summary

### List All Trials

```bash
python view_training_summary.py
```

This shows a table of all training runs with:
- Trial name
- Number of epochs completed
- Best accuracy achieved
- Latest loss and accuracy
- Last update time

### View Detailed Trial Information

```bash
python view_training_summary.py <trial_name>
```

This shows:
- General information (epochs, parameters, last update)
- Latest metrics
- Best performance
- Available checkpoints
- Training progress table for last 10 epochs

---

## 5. Log Files

Text logs are saved to: `outputs/<trial_name>/log.txt`

Each line is a JSON object containing:
```json
{
    "train_loss": 4.1234,
    "train_lr": 0.0005,
    "test_loss": 3.9876,
    "test_acc1": 12.34,
    "test_acc5": 34.56,
    "epoch": 0,
    "max_accuracy": 12.34,
    "n_parameters": 21346657
}
```

You can parse these logs programmatically for custom analysis.

---

## Example Workflow

### Starting Training
```bash
bash run_suit_training.sh \
    --batch-size 16 \
    --epochs 100 \
    --num-workers 4 \
    --model suit_small_224 \
    --trial-name sunrgbd_geolexels_v1 \
    --device cuda
```

### Monitoring (in another terminal)
```bash
# Option 1: Watch with Python script
python monitor_training.py sunrgbd_geolexels_v1 10

# Option 2: Launch TensorBoard
bash launch_tensorboard.sh sunrgbd_geolexels_v1

# Option 3: View summary
python view_training_summary.py sunrgbd_geolexels_v1
```

### After Training
```bash
# Compare all trials
python view_training_summary.py

# Check specific trial details
python view_training_summary.py sunrgbd_geolexels_v1
```

---

## Checkpoint Files

Checkpoints are saved in: `outputs/<trial_name>/`

- `checkpoint.pth` - Latest checkpoint (saved every epoch)
- `best_checkpoint.pth` - Best checkpoint (based on validation accuracy)

Each checkpoint contains:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Current epoch
- Model EMA state (if enabled)
- Loss scaler state
- Training arguments

---

## Troubleshooting

### TensorBoard not showing data
- Make sure training has started and at least one epoch is logged
- Check that the log directory exists: `logs/<trial_name>/`
- Try refreshing the browser

### Monitor script shows "Waiting for log file"
- Training hasn't started yet or the trial name is wrong
- Check that the output directory exists: `outputs/<trial_name>/`

### Progress bars not showing
- Make sure `tqdm` is installed: `pip install tqdm`
- Progress bars are disabled in non-main processes during distributed training

---

## Tips

1. **Use meaningful trial names:** Include model type, dataset, and version
   - Good: `sunrgbd_geolexels_small_v2`
   - Bad: `test123`

2. **Monitor GPU usage:** Use `nvidia-smi` or `watch -n 1 nvidia-smi`

3. **TensorBoard smoothing:** Adjust smoothing slider in TensorBoard UI for clearer trends

4. **Compare trials:** TensorBoard can display multiple trials simultaneously

5. **Resume training:** Use `--resume outputs/<trial_name>/checkpoint.pth` to continue from a checkpoint

---

## Metrics Explained

- **Loss:** Lower is better. Measures prediction error.
- **Acc@1:** Top-1 accuracy. Percentage where the top prediction is correct.
- **Acc@5:** Top-5 accuracy. Percentage where correct class is in top 5 predictions.
- **Learning Rate (LR):** Current learning rate. Typically decreases over time.
- **Grad Norm:** Gradient magnitude. Very high values indicate instability.

---

## Advanced: Custom Logging

To add custom metrics, edit [engine.py](engine.py):

```python
# In train_one_epoch():
if logger is not None and utils.is_main_process():
    logger.add_scalar('custom/my_metric', my_value, global_step)
```

Then view in TensorBoard under the "custom" namespace.
