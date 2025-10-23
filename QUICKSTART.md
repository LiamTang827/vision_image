# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have pip, install it first:
```bash
# On macOS
python3 -m ensurepip --upgrade

# On Linux
sudo apt-get install python3-pip

# On Windows
python -m ensurepip --upgrade
```

## Step 2: Verify Setup

```bash
python test_setup.py
```

This will check:
- ✓ All dependencies are installed
- ✓ Data folders exist
- ✓ Query images and bounding boxes can be loaded
- ✓ ResNet50 model can be loaded

## Step 3: Run Instance Search

```bash
python instance_search.py
```

**Expected Runtime:**
- With GPU: ~1-2 hours
- CPU only: ~15-20 hours

The script will:
1. Process all 50 queries with Method 1 (ResNet50)
2. Process all 50 queries with Method 2 (SIFT+Histogram)
3. Generate ranked lists for submission
4. Create visualizations for queries 0-4

## Step 4: Check Output

After completion, you should see:

```
✓ rankList_ResNet50.txt
✓ rankList_SIFT_Histogram.txt
✓ results_query_0_ResNet50.png
✓ results_query_1_ResNet50.png
✓ results_query_2_ResNet50.png
✓ results_query_3_ResNet50.png
✓ results_query_4_ResNet50.png
```

## Step 5: Prepare Submission

1. Convert REPORT.md to PDF (optional):
   ```bash
   # Using pandoc
   pandoc REPORT.md -o REPORT.pdf
   
   # Or copy content to Google Docs/Word and export as PDF
   ```

2. Create submission ZIP:
   ```bash
   zip -r submission.zip \
       instance_search.py \
       REPORT.pdf \
       rankList_ResNet50.txt \
       rankList_SIFT_Histogram.txt \
       results_query_0_ResNet50.png \
       results_query_1_ResNet50.png \
       results_query_2_ResNet50.png \
       results_query_3_ResNet50.png \
       results_query_4_ResNet50.png
   ```

## Troubleshooting

### Problem: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision
```

### Problem: "No module named 'cv2'"
**Solution:**
```bash
pip install opencv-python
```

### Problem: CUDA out of memory
**Solution:** Edit `instance_search.py`, line ~20:
```python
self.device = torch.device('cpu')  # Force CPU usage
```

### Problem: Process is too slow
**Solution:** 
- Use GPU if available
- Process fewer queries for testing:
  ```python
  # In main(), change:
  for query_id in range(5):  # Instead of range(50)
  ```

### Problem: Gallery folder not found
**Solution:** Ensure folder structure:
```
/Users/tangliam/Cityu Course/ vison and image/
├── gallery/
│   ├── gallery/  (28,493 images)
│   └── query/    (50 images)
└── query_txt/    (50 .txt files)
```

## Tips

1. **Test First**: Run on 1-2 queries before processing all 50
2. **Save Progress**: The code can be modified to save intermediate results
3. **Check Logs**: Monitor console output for any errors
4. **GPU Usage**: Check with `nvidia-smi` (Linux) or Activity Monitor (macOS)

## Need Help?

Contact TA: renweyang2-c@my.cityu.edu.hk
