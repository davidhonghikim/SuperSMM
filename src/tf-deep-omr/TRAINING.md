# Training Guide for tf-deep-omr

This document provides instructions for training the OMR model using the enhanced training scripts.

## Basic Training

The original training script `ctc_training.py` can be used for a single continuous training run:

```bash
python ctc_training.py -semantic -corpus Data/primus -set Data/train.txt -vocabulary Data/vocabulary_semantic.txt -save_model Models/trained_semantic_model/model
```

## Batch Training

For more flexibility, especially for long training sessions, use the batch training system:

### Features

- Train in small batches (default: 10 epochs per batch)
- Save checkpoints after each batch
- Resume training from the latest checkpoint
- Monitor training progress with detailed logs
- Visualize training metrics

### Commands

**Start a new training session:**

```bash
python batch_training.py -semantic \
    -corpus Data/primus \
    -set Data/train.txt \
    -vocabulary Data/vocabulary_semantic.txt \
    -save_model Models/batch_trained_model/model \
    -batch_size 10 \
    -total_epochs 50
```

**Resume training automatically:**

```bash
python batch_training.py -auto_resume -semantic \
    -corpus Data/primus \
    -set Data/train.txt \
    -vocabulary Data/vocabulary_semantic.txt \
    -save_model Models/batch_trained_model/model \
    -total_epochs 50
```

**Use the convenience script:**

```bash
./train_in_batches.sh
```

## Training Management

The `training_manager.py` script provides utilities for managing training:

**Find the latest checkpoint:**

```bash
python training_manager.py find_checkpoint -model_dir Models/batch_trained_model
```

**Analyze training logs:**

```bash
python training_manager.py analyze -log_file Models/batch_trained_model/training_log.json
```

**Plot training progress:**

```bash
python training_manager.py plot -log_file Models/batch_trained_model/training_log.json -output training_plot.png
```

**Generate resume command:**

```bash
python training_manager.py resume -model_dir Models/batch_trained_model \
    -corpus Data/primus \
    -set Data/train.txt \
    -vocabulary Data/vocabulary_semantic.txt \
    -semantic
```

## Visualization

To visualize training progress from log files:

```bash
python plot_training.py -log Models/batch_trained_model/training_log.json -output training_plot.png
```

## Prediction

After training, use the prediction script to recognize music symbols in images:

```bash
python ctc_predict.py -image Data/Example/000051652-1_2_1.png \
    -model Models/batch_trained_model/model-40.meta \
    -vocabulary Data/vocabulary_semantic.txt
```

For batch prediction on multiple images:

```bash
python batch_predict.py -model Models/batch_trained_model/model-40.meta \
    -vocabulary Data/vocabulary_semantic.txt \
    -image_dir Data/Example \
    -output results.json
```

## Tips for Better Training

1. **Start small**: Begin with 10-20 epochs to verify everything works
2. **Monitor loss**: Watch for decreasing loss values
3. **Validate regularly**: Check Symbol Error Rate (SER) on validation data
4. **Save checkpoints**: Use batch training to save progress
5. **Increase epochs**: For better results, train for 100+ epochs
6. **Adjust batch size**: Use smaller batch sizes (5-10) for more frequent checkpoints
7. **Use visualization**: Monitor training progress with plots
