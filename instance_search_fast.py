import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
import time

class GalleryDataset(Dataset):
    """Dataset for parallel loading of gallery images"""
    def __init__(self, gallery_path, transform=None):
        self.gallery_path = Path(gallery_path)
        self.image_files = sorted([f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')],
                                  key=lambda x: int(x.split('.')[0]))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.gallery_path / img_name
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.zeros(3, 224, 224)
            
            img_id = int(img_name.split('.')[0])
            return img_tensor, img_id, img_name
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            return torch.zeros(3, 224, 224), -1, img_name


class FastInstanceSearchSystem:
    def __init__(self, gallery_path, query_path, query_txt_path, batch_size=64, num_workers=8):
        self.gallery_path = Path(gallery_path)
        self.query_path = Path(query_path)
        self.query_txt_path = Path(query_txt_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load ResNet50
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model = nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.resnet_model.to(self.device)
        self.resnet_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Index storage
        self.gallery_features = None
        self.gallery_ids = None
        self.gallery_names = None
        
        print("Model loaded successfully")
    
    def build_gallery_index(self, save_path='gallery_index.npz', force_rebuild=False):
        """Build offline index using DataLoader for parallel processing"""
        if not force_rebuild and os.path.exists(save_path):
            print(f"Loading existing index from {save_path}...")
            data = np.load(save_path, allow_pickle=True)
            self.gallery_features = data['features']
            self.gallery_ids = data['ids']
            self.gallery_names = data['names']
            print(f"Loaded index: {len(self.gallery_ids)} images")
            return
        
        print(f"Building gallery index with batch_size={self.batch_size}, num_workers={self.num_workers}...")
        print("This will use GPU acceleration and parallel data loading!")
        
        # Create dataset and dataloader
        dataset = GalleryDataset(self.gallery_path, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        
        all_features = []
        all_ids = []
        all_names = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_imgs, batch_ids, batch_names in tqdm(dataloader, desc="Extracting features"):
                batch_imgs = batch_imgs.to(self.device, non_blocking=True)
                
                # Extract features
                features = self.resnet_model(batch_imgs)
                features = features.squeeze(-1).squeeze(-1)
                
                # Normalize
                features = features.cpu().numpy()
                features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
                
                all_features.append(features)
                all_ids.extend(batch_ids.numpy())
                all_names.extend(batch_names)
        
        # Concatenate all features
        self.gallery_features = np.vstack(all_features)
        self.gallery_ids = np.array(all_ids)
        self.gallery_names = np.array(all_names)
        
        elapsed = time.time() - start_time
        print(f"\nIndex built in {elapsed:.2f}s ({len(self.gallery_ids)/elapsed:.1f} images/sec)")
        print(f"Feature shape: {self.gallery_features.shape}")
        
        # Save index
        np.savez_compressed(
            save_path,
            features=self.gallery_features,
            ids=self.gallery_ids,
            names=self.gallery_names
        )
        print(f"Index saved to {save_path}")
    
    def load_bounding_boxes(self, query_id):
        txt_file = self.query_txt_path / f"{query_id}.txt"
        boxes = []
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = list(map(int, line.split()))
                        if len(parts) == 4:
                            boxes.append(parts)
        
        return boxes
    
    def extract_roi_features(self, img, bbox):
        """Extract ResNet features from ROI"""
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet_model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        features = features / (np.linalg.norm(features) + 1e-8)
        return features
    
    def process_query_fast(self, query_id):
        """Fast online query using pre-built index"""
        if self.gallery_features is None:
            raise RuntimeError("Gallery index not built! Call build_gallery_index() first.")
        
        query_img_path = self.query_path / f"{query_id}.jpg"
        query_img = cv2.imread(str(query_img_path))
        
        if query_img is None:
            print(f"Failed to load query image: {query_id}")
            return []
        
        bboxes = self.load_bounding_boxes(query_id)
        if not bboxes:
            print(f"No bounding boxes found for query {query_id}")
            return []
        
        # Extract query features
        query_features = []
        for bbox in bboxes:
            feat = self.extract_roi_features(query_img, bbox)
            if feat is not None:
                query_features.append(feat)
        
        if not query_features:
            print(f"No valid features for query {query_id}")
            return []
        
        # Fast similarity computation using vectorized operations
        query_features = np.array(query_features)  # (num_rois, feature_dim)
        
        # Compute similarities for all gallery images at once
        # Shape: (num_rois, num_gallery)
        similarities = query_features @ self.gallery_features.T
        
        # Take max similarity across all ROIs for each gallery image
        max_similarities = np.max(similarities, axis=0)
        
        # Get sorted indices
        sorted_indices = np.argsort(-max_similarities)
        ranked_list = self.gallery_ids[sorted_indices].tolist()
        
        return ranked_list
    
    def visualize_results(self, query_id, ranked_list, method_name, top_k=10):
        query_img_path = self.query_path / f"{query_id}.jpg"
        query_img = cv2.imread(str(query_img_path))
        bboxes = self.load_bounding_boxes(query_id)
        
        fig = plt.figure(figsize=(20, 12))
        
        # Show query with bounding boxes
        ax = plt.subplot(3, 4, 1)
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        ax.imshow(query_img_rgb)
        ax.set_title(f'Query {query_id}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        for bbox in bboxes:
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # Show top results
        for i, img_id in enumerate(ranked_list[:top_k]):
            ax = plt.subplot(3, 4, i + 2)
            img_path = self.gallery_path / f"{img_id}.jpg"
            img = cv2.imread(str(img_path))
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f'Rank {i+1}: {img_id}', fontsize=12)
            else:
                ax.text(0.5, 0.5, f'Image {img_id}\nnot found', ha='center', va='center')
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results_query_{query_id}_{method_name}.png', dpi=150, bbox_inches='tight')
        plt.close()


class FastHybridFeatureExtractor:
    """Fast SIFT + Color Histogram with parallel processing"""
    
    def __init__(self, gallery_path, query_path, query_txt_path, num_workers=8):
        self.gallery_path = Path(gallery_path)
        self.query_path = Path(query_path)
        self.query_txt_path = Path(query_txt_path)
        self.num_workers = num_workers
        
        self.gallery_features = None
        self.gallery_ids = None
        self.gallery_names = None
        
        print("Fast Hybrid extractor ready")
    
    def extract_hybrid_features(self, img_path):
        """Extract hybrid features from image path"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None
        
        # SIFT part
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=50)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None and len(descriptors) > 0:
            sift_feat = np.mean(descriptors, axis=0)
            sift_feat = sift_feat / (np.linalg.norm(sift_feat) + 1e-8)
        else:
            sift_feat = np.zeros(128)
        
        # Color histogram part
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-8)
        s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-8)
        v_hist = v_hist.flatten() / (np.sum(v_hist) + 1e-8)
        
        hist_feat = np.concatenate([h_hist, s_hist, v_hist])
        
        # Combine features
        hybrid_feat = np.concatenate([0.6 * sift_feat, 0.4 * hist_feat])
        hybrid_feat = hybrid_feat / (np.linalg.norm(hybrid_feat) + 1e-8)
        
        img_name = img_path.name
        img_id = int(img_name.split('.')[0])
        
        return hybrid_feat, img_id
    
    def build_gallery_index(self, save_path='gallery_index_hybrid.npz', force_rebuild=False):
        """Build offline index using parallel processing"""
        if not force_rebuild and os.path.exists(save_path):
            print(f"Loading existing hybrid index from {save_path}...")
            data = np.load(save_path, allow_pickle=True)
            self.gallery_features = data['features']
            self.gallery_ids = data['ids']
            self.gallery_names = data['names']
            print(f"Loaded hybrid index: {len(self.gallery_ids)} images")
            return
        
        print(f"Building hybrid gallery index with {self.num_workers} workers...")
        
        gallery_images = sorted([self.gallery_path / f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')],
                               key=lambda x: int(x.name.split('.')[0]))
        
        start_time = time.time()
        
        all_features = []
        all_ids = []
        all_names = []
        
        # Parallel feature extraction using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(self.extract_hybrid_features, gallery_images),
                total=len(gallery_images),
                desc="Extracting hybrid features"
            ))
        
        for feat, img_id in results:
            if feat is not None:
                all_features.append(feat)
                all_ids.append(img_id)
        
        self.gallery_features = np.array(all_features)
        self.gallery_ids = np.array(all_ids)
        
        elapsed = time.time() - start_time
        print(f"\nHybrid index built in {elapsed:.2f}s ({len(self.gallery_ids)/elapsed:.1f} images/sec)")
        print(f"Feature shape: {self.gallery_features.shape}")
        
        # Save index
        np.savez_compressed(
            save_path,
            features=self.gallery_features,
            ids=self.gallery_ids,
            names=self.gallery_names
        )
        print(f"Hybrid index saved to {save_path}")
    
    def load_bounding_boxes(self, query_id):
        txt_file = self.query_txt_path / f"{query_id}.txt"
        boxes = []
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = list(map(int, line.split()))
                        if len(parts) == 4:
                            boxes.append(parts)
        
        return boxes
    
    def extract_roi_hybrid_features(self, img, bbox):
        """Extract hybrid features from ROI"""
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # SIFT part
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=50)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None and len(descriptors) > 0:
            sift_feat = np.mean(descriptors, axis=0)
            sift_feat = sift_feat / (np.linalg.norm(sift_feat) + 1e-8)
        else:
            sift_feat = np.zeros(128)
        
        # Color histogram part
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-8)
        s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-8)
        v_hist = v_hist.flatten() / (np.sum(v_hist) + 1e-8)
        
        hist_feat = np.concatenate([h_hist, s_hist, v_hist])
        
        # Combine features
        hybrid_feat = np.concatenate([0.6 * sift_feat, 0.4 * hist_feat])
        hybrid_feat = hybrid_feat / (np.linalg.norm(hybrid_feat) + 1e-8)
        
        return hybrid_feat
    
    def process_query_fast(self, query_id):
        """Fast online query using pre-built index"""
        if self.gallery_features is None:
            raise RuntimeError("Gallery index not built! Call build_gallery_index() first.")
        
        query_img_path = self.query_path / f"{query_id}.jpg"
        query_img = cv2.imread(str(query_img_path))
        
        if query_img is None:
            return []
        
        bboxes = self.load_bounding_boxes(query_id)
        if not bboxes:
            return []
        
        query_features = []
        for bbox in bboxes:
            feat = self.extract_roi_hybrid_features(query_img, bbox)
            if feat is not None:
                query_features.append(feat)
        
        if not query_features:
            return []
        
        # Fast vectorized similarity computation
        query_features = np.array(query_features)
        similarities = query_features @ self.gallery_features.T
        max_similarities = np.max(similarities, axis=0)
        
        sorted_indices = np.argsort(-max_similarities)
        ranked_list = self.gallery_ids[sorted_indices].tolist()
        
        return ranked_list


def main():
    gallery_path = "gallery/gallery"
    query_path = "gallery/query"
    query_txt_path = "query_txt"
    
    print("="*60)
    print("FAST INSTANCE SEARCH - GPU ACCELERATED")
    print("="*60)
    
    # Method 1: ResNet50 with GPU batch processing
    print("\n[METHOD 1] ResNet50 Deep Features")
    print("-"*60)
    
    system_resnet = FastInstanceSearchSystem(
        gallery_path, query_path, query_txt_path,
        batch_size=128,  # Increased batch size for RTX 3070
        num_workers=8    # Parallel data loading
    )
    
    # Build index (offline phase)
    system_resnet.build_gallery_index('gallery_index_resnet.npz', force_rebuild=False)
    
    # Online query phase
    print("\nProcessing queries...")
    rankList_method1 = []
    query_start = time.time()
    
    for query_id in tqdm(range(50), desc="ResNet50 queries"):
        ranked_list = system_resnet.process_query_fast(query_id)
        rankList_method1.append(ranked_list)
        
        if query_id < 5:
            system_resnet.visualize_results(query_id, ranked_list, 'ResNet50_Fast', top_k=10)
    
    query_elapsed = time.time() - query_start
    print(f"All 50 queries completed in {query_elapsed:.2f}s ({query_elapsed/50:.3f}s per query)")
    
    # Save results
    with open('rankList_ResNet50.txt', 'w') as f:
        for i, ranked_list in enumerate(rankList_method1):
            f.write(f"Q{i+1}: " + " ".join(map(str, ranked_list)) + "\n")
    
    print("Saved rankList_ResNet50.txt")
    
    # Method 2: SIFT + Histogram with parallel processing
    print("\n[METHOD 2] SIFT + Color Histogram")
    print("-"*60)
    
    system_hybrid = FastHybridFeatureExtractor(
        gallery_path, query_path, query_txt_path,
        num_workers=8
    )
    
    # Build index (offline phase)
    system_hybrid.build_gallery_index('gallery_index_hybrid.npz', force_rebuild=False)
    
    # Online query phase
    print("\nProcessing queries...")
    rankList_method2 = []
    query_start = time.time()
    
    for query_id in tqdm(range(50), desc="Hybrid queries"):
        ranked_list = system_hybrid.process_query_fast(query_id)
        rankList_method2.append(ranked_list)
    
    query_elapsed = time.time() - query_start
    print(f"All 50 queries completed in {query_elapsed:.2f}s ({query_elapsed/50:.3f}s per query)")
    
    # Save results
    with open('rankList_SIFT_Histogram.txt', 'w') as f:
        for i, ranked_list in enumerate(rankList_method2):
            f.write(f"Q{i+1}: " + " ".join(map(str, ranked_list)) + "\n")
    
    print("Saved rankList_SIFT_Histogram.txt")
    
    print("\n" + "="*60)
    print("ALL DONE! Check the output files and visualizations.")
    print("="*60)


if __name__ == "__main__":
    main()
