# Instance Search Assignment

## Overview
This project implements an instance search system for retrieving similar images from a large gallery based on query images with bounding box annotations.

## Dataset Structure
```
.
├── gallery/
│   ├── gallery/          # 28,493 gallery images (0.jpg - 28492.jpg)
│   └── query/            # 50 query images (0.jpg - 49.jpg)
├── query_txt/            # Bounding box annotations (0.txt - 49.txt)
├── instance_search.py    # Main implementation
├── REPORT.md             # Detailed report
└── README.md             # This file
```

## Requirements

### Python Packages
Install required packages:
```bash
pip install torch torchvision opencv-python numpy pillow matplotlib tqdm
```

### Detailed Requirements
- Python 3.7+
- PyTorch 1.8+ (with CUDA support optional but recommended)
- torchvision
- opencv-python (cv2)
- numpy
- pillow (PIL)
- matplotlib
- tqdm

## Methods Implemented

### Method 1: ResNet50 Deep Features
- Uses pre-trained ResNet50 CNN for feature extraction
- Extracts 2048-dimensional features from ROIs
- Computes cosine similarity for ranking
- **Best for**: Semantic similarity, object-level matching
- **Requires**: GPU recommended (but works on CPU)

### Method 2: SIFT + Color Histogram Fusion
- Combines SIFT local features with HSV color histograms
- Creates 224-dimensional hybrid features (128-D SIFT + 96-D histogram)
- Weighted fusion (60% SIFT, 40% color)
- **Best for**: Texture and color-based matching
- **Requires**: CPU only

## Usage

### Quick Start
```bash
# Run the complete pipeline
python instance_search.py
```

This will:
1. Process all 50 queries using both methods
2. Generate ranked lists for 28,493 gallery images
3. Create visualizations for queries 1-5
4. Save output files

### Expected Output Files

**Rank Lists** (submission files):
- `rankList_ResNet50.txt` - Method 1 results
- `rankList_SIFT_Histogram.txt` - Method 2 results

**Visualizations** (for queries 0-4):
- `results_query_0_ResNet50.png`
- `results_query_1_ResNet50.png`
- `results_query_2_ResNet50.png`
- `results_query_3_ResNet50.png`
- `results_query_4_ResNet50.png`

### Rank List Format
Each file contains 50 lines (one per query):
```
Q1: 7 12 214 350 ... (all 28,493 image IDs in descending order of similarity)
Q2: 301 501 1990 2 ...
Q3: 288 345 389 1290 ...
...
Q50: ...
```

## Bounding Box Format

Query bounding boxes are stored in `query_txt/*.txt` files:
```
x y width height
x y width height  # (if there are two objects)
```

Example (`query_txt/0.txt`):
```
528 98 284 497
230 184 353 353
```

## Processing Time Estimates

### Method 1 (ResNet50)
- **With GPU**: ~1-2 hours for all 50 queries
- **CPU only**: ~15-20 hours for all 50 queries

### Method 2 (SIFT + Histogram)
- **CPU**: ~8-10 hours for all 50 queries

## Advanced Usage

### Process Single Query
```python
from instance_search import InstanceSearchSystem

system = InstanceSearchSystem("gallery/gallery", "gallery/query", "query_txt")
ranked_list = system.process_query(query_id=0, method='resnet')
print(f"Top 10 results: {ranked_list[:10]}")
```

### Custom Feature Extraction
```python
# Extract ResNet features from custom ROI
bbox = [100, 100, 200, 200]  # x, y, w, h
img = cv2.imread("image.jpg")
features = system.extract_roi_features(img, bbox, method='resnet')
```

## Troubleshooting

### CUDA Out of Memory
If you get GPU memory errors:
1. Reduce batch size (process images one at a time)
2. Use CPU instead: Set `device = torch.device('cpu')`
3. Close other GPU applications

### Missing Dependencies
```bash
# Install all requirements
pip install -r requirements.txt
```

### Image Loading Errors
Ensure all paths are correct:
- Gallery images: `gallery/gallery/*.jpg`
- Query images: `gallery/query/*.jpg`
- Query text files: `query_txt/*.txt`

## Performance Tips

1. **Use GPU**: ~10-15x faster for ResNet50 method
2. **Parallel Processing**: Modify code to process multiple galleries in parallel
3. **Feature Caching**: Save extracted features to disk for repeated experiments
4. **Smaller Resolution**: Resize gallery images if exact pixel accuracy isn't needed

## Submission Checklist

✓ Two implementation methods (one deep learning-based)  
✓ Python code (`instance_search.py`)  
✓ Report (1-6 pages) - `REPORT.md` or convert to PDF  
✓ Rank List 1: `rankList_ResNet50.txt`  
✓ Rank List 2: `rankList_SIFT_Histogram.txt`  
✓ Visualizations for queries 1-5  

## File Structure for Submission

Create a ZIP file with:
```
submission.zip
├── instance_search.py        # Main code
├── REPORT.pdf                 # Report (convert REPORT.md to PDF)
├── rankList_ResNet50.txt      # Method 1 results
├── rankList_SIFT_Histogram.txt # Method 2 results
├── results_query_0_ResNet50.png
├── results_query_1_ResNet50.png
├── results_query_2_ResNet50.png
├── results_query_3_ResNet50.png
├── results_query_4_ResNet50.png
└── README.md                  # This file (optional)
```

## Notes

- Image IDs in rank lists should NOT include `.jpg` extension
- Image IDs are integers from 0 to 28492
- Ensure all 28,493 images appear in each ranked list
- Query numbering: Q1 corresponds to `0.jpg`, Q2 to `1.jpg`, etc.

## Contact

For technical questions, email the Teaching Assistant:
- **Renwei Yang**: renweyang2-c@my.cityu.edu.hk

## License

This code is for educational purposes only (CityU Course Assignment).
