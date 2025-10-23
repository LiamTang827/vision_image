"""
Test and validate the instance search system setup
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def check_data_structure():
    """Check if data folders and files exist"""
    print("\nChecking data structure...")
    
    # Check folders
    gallery_path = Path("gallery/gallery")
    query_path = Path("gallery/query")
    query_txt_path = Path("query_txt")
    
    if not gallery_path.exists():
        print(f"✗ Gallery folder not found: {gallery_path}")
        return False
    else:
        gallery_count = len([f for f in os.listdir(gallery_path) if f.endswith('.jpg')])
        print(f"✓ Gallery folder found with {gallery_count} images (expected: 28,493)")
    
    if not query_path.exists():
        print(f"✗ Query folder not found: {query_path}")
        return False
    else:
        query_count = len([f for f in os.listdir(query_path) if f.endswith('.jpg')])
        print(f"✓ Query folder found with {query_count} images (expected: 50)")
    
    if not query_txt_path.exists():
        print(f"✗ Query text folder not found: {query_txt_path}")
        return False
    else:
        txt_count = len([f for f in os.listdir(query_txt_path) if f.endswith('.txt')])
        print(f"✓ Query text folder found with {txt_count} files (expected: 50)")
    
    return True

def test_query_loading():
    """Test loading a query image and bounding boxes"""
    print("\nTesting query loading...")
    
    try:
        import cv2
        from pathlib import Path
        
        # Test query 0
        query_img_path = Path("gallery/query/0.jpg")
        query_txt_path = Path("query_txt/0.txt")
        
        if not query_img_path.exists():
            print(f"✗ Query image not found: {query_img_path}")
            return False
        
        img = cv2.imread(str(query_img_path))
        if img is None:
            print(f"✗ Failed to load query image: {query_img_path}")
            return False
        
        print(f"✓ Loaded query image 0: shape {img.shape}")
        
        # Load bounding boxes
        if not query_txt_path.exists():
            print(f"✗ Query text file not found: {query_txt_path}")
            return False
        
        with open(query_txt_path, 'r') as f:
            lines = f.readlines()
            print(f"✓ Loaded {len(lines)} bounding box(es) from query 0")
            for i, line in enumerate(lines):
                parts = list(map(int, line.strip().split()))
                if len(parts) == 4:
                    x, y, w, h = parts
                    print(f"  Box {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def test_model_loading():
    """Test loading pre-trained models"""
    print("\nTesting model loading...")
    
    try:
        import torch
        from torchvision import models
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, will use CPU (slower)")
        
        print("Loading ResNet50...")
        model = models.resnet50(pretrained=True)
        print("✓ ResNet50 loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def main():
    print("="*60)
    print("Instance Search System - Setup Validation")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete - install missing dependencies")
        sys.exit(1)
    
    # Check data structure
    if not check_data_structure():
        print("\n❌ Data structure incomplete - ensure all folders exist")
        sys.exit(1)
    
    # Test query loading
    if not test_query_loading():
        print("\n❌ Query loading failed")
        sys.exit(1)
    
    # Test model loading
    if not test_model_loading():
        print("\n⚠ Model loading had issues, but system may still work")
    
    print("\n" + "="*60)
    print("✓ System validation complete!")
    print("="*60)
    print("\nYou can now run:")
    print("  python instance_search.py")
    print("\nThis will process all 50 queries and generate:")
    print("  - rankList_ResNet50.txt")
    print("  - rankList_SIFT_Histogram.txt")
    print("  - Visualization images for queries 0-4")

if __name__ == "__main__":
    main()
