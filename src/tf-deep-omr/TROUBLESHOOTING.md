# OMR Training Troubleshooting Guide

This document provides solutions for common issues encountered during OMR model training.

## Missing Images in Corpus

### Symptoms
- Training stops with errors about missing image files
- Errors like: `FileNotFoundError: [Errno 2] No such file or directory: '...'`

### Solutions
1. Run the corpus preparation tool to check your corpus structure:
   ```bash
   python prepare_corpus.py -corpus "./Data" -set "./data/train.txt"
   ```

2. If missing images are reported, fix the paths:
   ```bash
   python prepare_corpus.py -corpus "./Data" -set "./data/train.txt" -fix
   ```

3. Update your training script to use the fixed set file:
   ```bash
   # In train_with_recovery.sh
   SET="./data/train_fixed.txt"
   ```

## Out of Memory Errors

### Symptoms
- Training crashes with CUDA out of memory errors
- System becomes unresponsive during training

### Solutions
1. Reduce batch size in your training script:
   ```bash
   # In train_with_recovery.sh
   BATCH_SIZE=5  # Try smaller values like 2 or 1
   ```

2. Enable memory growth to prevent TensorFlow from allocating all GPU memory:
   ```bash
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

3. Close other GPU-intensive applications during training

## Training Too Slow

### Symptoms
- Each epoch takes a very long time to complete
- Progress seems much slower than expected

### Solutions
1. Enable TensorFlow optimizations in your training script:
   ```bash
   export TF_ENABLE_AUTO_MIXED_PRECISION=1
   export TF_GPU_THREAD_MODE=gpu_private
   ```

2. Use a smaller dataset for testing:
   ```bash
   ./create_sample_dataset.sh
   ```

3. Monitor system resources to identify bottlenecks:
   ```bash
   python monitor_training.py -model_dir "./model/semantic_model" -live
   ```

## Training Not Resuming After Crash

### Symptoms
- Training starts from epoch 0 after a crash instead of resuming
- Checkpoints not being loaded

### Solutions
1. Check if checkpoint files exist in your model directory:
   ```bash
   ls -la ./model/semantic_model/
   ```

2. Verify the recovery file exists and is valid:
   ```bash
   cat ./model/semantic_model/recovery.json
   ```

3. Manually specify the checkpoint to load:
   ```bash
   python auto_recover_training.py ... -load_checkpoint "./model/semantic_model/model-X"
   ```
   Replace X with the epoch number of the checkpoint you want to load.

## Email Notifications Not Working

### Symptoms
- No email notifications received during training
- No errors related to email sending in logs

### Solutions
1. Check your SMTP settings in the training script:
   ```bash
   # In train_production.sh
   EMAIL_TO="your.email@example.com"
   SMTP_SERVER="smtp.example.com"
   SMTP_PORT=587
   ```

2. For Gmail, you may need to enable "Less secure app access" or use an App Password

3. Test email sending with a simple Python script:
   ```python
   import smtplib
   from email.message import EmailMessage
   
   msg = EmailMessage()
   msg.set_content("Test email")
   msg["Subject"] = "OMR Training Test"
   msg["From"] = "sender@example.com"
   msg["To"] = "recipient@example.com"
   
   s = smtplib.SMTP("smtp.example.com", 587)
   s.starttls()
   s.login("username", "password")
   s.send_message(msg)
   s.quit()
   ```

## Still Having Issues?

If you're still experiencing problems:

1. Check the detailed logs in `./model/semantic_model/logs/`
2. Run a test training with minimal epochs:
   ```bash
   ./test_training.sh
   ```
3. Try with a completely fresh model directory:
   ```bash
   rm -rf ./model/semantic_model
   mkdir -p ./model/semantic_model
   ```
