# TensorFlow Deep OMR

This is a TensorFlow implementation of an Optical Music Recognition (OMR) system using Connectionist Temporal Classification (CTC) loss.

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-tf-deep-omr.txt
   ```

## Training

### Quick Start

To train the model with default settings:

```bash
python src/tf-deep-omr/train_with_logging.py --config src/tf-deep-omr/config/training_config.yaml
```

### Configuration

Edit `src/tf-deep-omr/config/training_config.yaml` to customize training parameters:

```yaml
# Data parameters
corpus: ./src/Data/primus
set_file: ./src/Data/train.txt
vocabulary: ./src/Data/vocabulary_semantic.txt

# Model parameters
img_height: 128
batch_size: 50
epochs: 50
learning_rate: 0.001

# Logging
log_dir: ./logs
checkpoint_dir: ./checkpoints
```

### Resuming Training

To resume training from the latest checkpoint:

```bash
python src/tf-deep-omr/train_with_logging.py --resume
```

## Logging

Training logs are stored in the following locations:

- **Console Output**: Real-time training progress
- **CSV Logs**: `logs/training_metrics_<timestamp>.csv`
  - Contains detailed metrics for each epoch (loss, time, etc.)
- **TensorBoard Logs**: `logs/tensorboard/`
  - Visualize training metrics with TensorBoard

To view TensorBoard:

```bash
tensorboard --logdir=logs/tensorboard
```

## Monitoring

### Performance Metrics

- GPU/CPU usage
- Memory consumption
- Batch processing time

### Error Handling

- Automatic error logging
- Checkpoint saving on error
- Graceful shutdown

## Model Architecture

The model uses a CNN-RNN architecture with CTC loss:

1. **Feature Extraction**: Multiple CNN blocks
2. **Sequence Modeling**: Bidirectional LSTM layers
3. **Output**: CTC loss for sequence prediction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
