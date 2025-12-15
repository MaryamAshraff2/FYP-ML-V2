# core/reid/feature_extractor.py
"""
Feature Extractor - Extract ReID embeddings from person crops
Based on your working v1 code with OSNet/ResNet50
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import List, Optional
import torchreid

# Global model cache to avoid reloading
_REID_MODEL = None
_REID_TRANSFORM = None


def init_reid_model(model_name: str = 'osnet_x1_0', device: str = 'cpu'):
    """
    Initialize the ReID model (OSNet or ResNet50).
    
    Args:
        model_name: Model architecture ('osnet_x1_0', 'osnet_ain_x1_0', 'resnet50')
        device: 'cpu' or 'cuda'
        
    Returns:
        Loaded model ready for inference
    """
    global _REID_MODEL, _REID_TRANSFORM
    
    print(f"🔧 Loading ReID model: {model_name}...")
    
    try:
        # Load pretrained model from torchreid
        model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,  # Will be ignored for feature extraction
            pretrained=True,
            use_gpu=(device == 'cuda')
        )
        
        model.eval()
        model = model.to(device)
        
        # Define preprocessing transform
        _REID_TRANSFORM = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        _REID_MODEL = model
        print(f"✅ ReID model loaded: {model_name}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading ReID model: {e}")
        raise


def get_reid_model():
    """Get the global ReID model instance."""
    global _REID_MODEL
    if _REID_MODEL is None:
        _REID_MODEL = init_reid_model()
    return _REID_MODEL


def get_reid_transform():
    """Get the global transform instance."""
    global _REID_TRANSFORM
    if _REID_TRANSFORM is None:
        init_reid_model()  # Will initialize both model and transform
    return _REID_TRANSFORM


def crop_person(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """
    Crop person region from frame.
    
    Args:
        frame: Input frame (BGR)
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Cropped person image or None if invalid
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Check if bbox is valid
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop
        crop = frame[y1:y2, x1:x2]
        
        # Check if crop is too small
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        return crop
        
    except Exception as e:
        print(f"⚠️ Error cropping person: {e}")
        return None


def extract_reid_embedding(
    frame: np.ndarray, 
    bbox: List[float],
    model: Optional[torch.nn.Module] = None
) -> Optional[np.ndarray]:
    """
    Extract ReID embedding from a person crop.
    
    Args:
        frame: Input frame (BGR)
        bbox: Bounding box [x1, y1, x2, y2]
        model: ReID model (will use global if None)
        
    Returns:
        512-dim normalized embedding vector or None if extraction fails
    """
    try:
        # Get model and transform
        if model is None:
            model = get_reid_model()
        transform = get_reid_transform()
        
        # Crop person
        crop = crop_person(frame, bbox)
        if crop is None:
            return None
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        input_tensor = transform(crop_rgb).unsqueeze(0)
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(input_tensor)
        
        # Normalize embedding (L2 normalization)
        embedding = F.normalize(features, p=2, dim=1)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()[0]
        
        return embedding_np
        
    except Exception as e:
        print(f"⚠️ Error extracting embedding: {e}")
        return None


def extract_batch_embeddings(
    frame: np.ndarray,
    bboxes: List[List[float]],
    model: Optional[torch.nn.Module] = None,
    batch_size: int = 8
) -> List[Optional[np.ndarray]]:
    """
    Extract embeddings for multiple persons in batch (faster).
    
    Args:
        frame: Input frame (BGR)
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        model: ReID model (will use global if None)
        batch_size: Batch size for inference
        
    Returns:
        List of normalized embeddings (same length as bboxes, None for failed crops)
    """
    if not bboxes:
        return []
    
    try:
        # Get model and transform
        if model is None:
            model = get_reid_model()
        transform = get_reid_transform()
        device = next(model.parameters()).device
        
        embeddings = []
        valid_indices = []
        batch_tensors = []
        
        # Prepare crops
        for idx, bbox in enumerate(bboxes):
            crop = crop_person(frame, bbox)
            if crop is None:
                embeddings.append(None)
                continue
            
            # Convert and transform
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(crop_rgb)
            
            batch_tensors.append(tensor)
            valid_indices.append(idx)
            embeddings.append(None)  # Placeholder
        
        # Process in batches
        if batch_tensors:
            for i in range(0, len(batch_tensors), batch_size):
                batch = torch.stack(batch_tensors[i:i+batch_size]).to(device)
                
                with torch.no_grad():
                    features = model(batch)
                    features = F.normalize(features, p=2, dim=1)
                
                # Store results
                features_np = features.cpu().numpy()
                for j, feat in enumerate(features_np):
                    orig_idx = valid_indices[i + j]
                    embeddings[orig_idx] = feat
        
        return embeddings
        
    except Exception as e:
        print(f"⚠️ Error in batch embedding extraction: {e}")
        return [None] * len(bboxes)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2 normalize an embedding vector.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        L2 normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


# Convenience function matching your v1 interface
def extract_embedding(frame: np.ndarray, bbox: List[float], model=None) -> Optional[np.ndarray]:
    """
    Alias for extract_reid_embedding to match v1 interface.
    
    Args:
        frame: Input frame (BGR)
        bbox: Bounding box [x1, y1, x2, y2]
        model: ReID model (optional)
        
    Returns:
        Normalized 512-dim embedding or None
    """
    return extract_reid_embedding(frame, bbox, model)