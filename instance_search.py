import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class InstanceSearchSystem:
    def __init__(self, gallery_path, query_path, query_txt_path):
        self.gallery_path = Path(gallery_path)
        self.query_path = Path(query_path)
        self.query_txt_path = Path(query_txt_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load ResNet50
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
        self.resnet_model.to(self.device)
        self.resnet_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("Model loaded")
    
    def load_bounding_boxes(self, query_id):
        # Read bounding box file
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
    
    def extract_roi_features(self, img, bbox, method='resnet'):
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        if method == 'resnet':
            return self.extract_resnet_features(roi)
        elif method == 'sift':
            return self.extract_sift_features(roi)
        elif method == 'histogram':
            return self.extract_histogram_features(roi)
    
    def extract_resnet_features(self, img):
        if img is None or img.size == 0:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet_model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        return features
    
    def extract_sift_features(self, img):
        if img is None or img.size == 0:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sift = cv2.SIFT_create(nfeatures=100)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128)
        
        feature = np.mean(descriptors, axis=0)
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature
    
    def extract_histogram_features(self, img):
        if img is None or img.size == 0:
            return None
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-8)
        s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-8)
        v_hist = v_hist.flatten() / (np.sum(v_hist) + 1e-8)
        
        feature = np.concatenate([h_hist, s_hist, v_hist])
        return feature
    
    def compute_similarity(self, feat1, feat2, method='cosine'):
        if feat1 is None or feat2 is None:
            return 0.0
        
        if method == 'cosine':
            similarity = np.dot(feat1, feat2)
            return max(0.0, similarity)
        elif method == 'euclidean':
            distance = np.linalg.norm(feat1 - feat2)
            return 1.0 / (1.0 + distance)
    
    def process_query(self, query_id, method='resnet'):
        query_img_path = self.query_path / f"{query_id}.jpg"
        query_img = cv2.imread(str(query_img_path))
        
        if query_img is None:
            print(f"Failed to load query image: {query_id}")
            return []
        
        bboxes = self.load_bounding_boxes(query_id)
        if not bboxes:
            print(f"No bounding boxes found for query {query_id}")
            return []
        
        # Extract features from query ROIs
        query_features = []
        for bbox in bboxes:
            feat = self.extract_roi_features(query_img, bbox, method=method)
            if feat is not None:
                query_features.append(feat)
        
        if not query_features:
            print(f"No valid features for query {query_id}")
            return []
        
        # Get gallery images
        gallery_images = sorted([f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')],
                               key=lambda x: int(x.split('.')[0]))
        
        similarities = []
        print(f"Processing query {query_id} ({len(gallery_images)} images)...")
        
        for img_name in tqdm(gallery_images, desc=f"Query {query_id}"):
            img_path = self.gallery_path / img_name
            img = cv2.imread(str(img_path))
            
            if img is None:
                similarities.append((img_name, 0.0))
                continue
            
            # Extract gallery features
            if method == 'resnet':
                gallery_feat = self.extract_resnet_features(img)
            elif method == 'sift':
                gallery_feat = self.extract_sift_features(img)
            elif method == 'histogram':
                gallery_feat = self.extract_histogram_features(img)
            else:
                gallery_feat = None
            
            if gallery_feat is None:
                similarities.append((img_name, 0.0))
                continue
            
            # Find max similarity across all query ROIs
            max_sim = 0.0
            for query_feat in query_features:
                sim = self.compute_similarity(query_feat, gallery_feat)
                max_sim = max(max_sim, sim)
            
            similarities.append((img_name, max_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        ranked_list = [int(img_name.split('.')[0]) for img_name, _ in similarities]
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
        print(f"Saved: results_query_{query_id}_{method_name}.png")


class HybridFeatureExtractor:
    # SIFT + Color Histogram method
    
    def __init__(self, gallery_path, query_path, query_txt_path):
        self.gallery_path = Path(gallery_path)
        self.query_path = Path(query_path)
        self.query_txt_path = Path(query_txt_path)
        print("Hybrid extractor ready")
    
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
    
    def extract_hybrid_features(self, img, bbox=None):
        if bbox is not None:
            x, y, w, h = bbox
            roi = img[y:y+h, x:x+w]
        else:
            roi = img
        
        if roi.size == 0:
            return None
        
        # SIFT part
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
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
    
    def process_query(self, query_id):
        query_img_path = self.query_path / f"{query_id}.jpg"
        query_img = cv2.imread(str(query_img_path))
        
        if query_img is None:
            return []
        
        bboxes = self.load_bounding_boxes(query_id)
        if not bboxes:
            return []
        
        query_features = []
        for bbox in bboxes:
            feat = self.extract_hybrid_features(query_img, bbox)
            if feat is not None:
                query_features.append(feat)
        
        if not query_features:
            return []
        
        gallery_images = sorted([f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')],
                               key=lambda x: int(x.split('.')[0]))
        
        similarities = []
        
        for img_name in tqdm(gallery_images, desc=f"Query {query_id}"):
            img_path = self.gallery_path / img_name
            img = cv2.imread(str(img_path))
            
            if img is None:
                similarities.append((img_name, 0.0))
                continue
            
            gallery_feat = self.extract_hybrid_features(img, bbox=None)
            
            if gallery_feat is None:
                similarities.append((img_name, 0.0))
                continue
            
            max_sim = 0.0
            for query_feat in query_features:
                sim = np.dot(query_feat, gallery_feat)
                max_sim = max(max_sim, sim)
            
            similarities.append((img_name, max_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        ranked_list = [int(img_name.split('.')[0]) for img_name, _ in similarities]
        
        return ranked_list


def main():
    # Setup paths
    gallery_path = "gallery/gallery"
    query_path = "gallery/query"
    query_txt_path = "query_txt"
    
    # Run Method 1: ResNet50 Deep Features
    print("\nStarting Method 1: ResNet50 Deep Features")
    
    system_resnet = InstanceSearchSystem(gallery_path, query_path, query_txt_path)
    rankList_method1 = []
    
    for query_id in range(50):
        ranked_list = system_resnet.process_query(query_id, method='resnet')
        rankList_method1.append(ranked_list)
        
        # Visualize first 5 queries
        if query_id < 5:
            system_resnet.visualize_results(query_id, ranked_list, 'ResNet50', top_k=10)
    
    # Save results
    with open('rankList_ResNet50.txt', 'w') as f:
        for i, ranked_list in enumerate(rankList_method1):
            f.write(f"Q{i+1}: " + " ".join(map(str, ranked_list)) + "\n")
    
    print("Saved rankList_ResNet50.txt")
    
    # Run Method 2: SIFT + Color Histogram
    print("\nStarting Method 2: SIFT + Color Histogram")
    
    system_hybrid = HybridFeatureExtractor(gallery_path, query_path, query_txt_path)
    rankList_method2 = []
    
    for query_id in range(50):
        ranked_list = system_hybrid.process_query(query_id)
        rankList_method2.append(ranked_list)
    
    # Save results
    with open('rankList_SIFT_Histogram.txt', 'w') as f:
        for i, ranked_list in enumerate(rankList_method2):
            f.write(f"Q{i+1}: " + " ".join(map(str, ranked_list)) + "\n")
    
    print("Saved rankList_SIFT_Histogram.txt")


if __name__ == "__main__":
    main()
