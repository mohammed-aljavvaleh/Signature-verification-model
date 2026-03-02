"""
Dataset loader for Signature Verification
Creates pairs of signatures (genuine-genuine and genuine-forged)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import random
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List
import os


class SignatureDataset(Dataset):
    """
    Dataset class for signature verification using Siamese Networks
    
    Creates pairs of signature images:
    - Positive pairs: (genuine, genuine) from same user -> label = 1
    - Negative pairs: (genuine, forged) or (genuine from different users) -> label = 0
    """
    
    def __init__(self, data_dir: Path, transform=None, pairs_per_user: int = 50):
        """
        Args:
            data_dir: Path to directory containing 'genuine' and 'forged' folders
            transform: Optional transforms to apply to images
            pairs_per_user: Number of pairs to generate per user
        """
        self.data_dir = data_dir
        self.transform = transform
        self.pairs_per_user = pairs_per_user
        
        # Load all signature images
        self.genuine_dir = data_dir / "genuine"
        self.forged_dir = data_dir / "forged"
        
        # Get all genuine signatures organized by user
        self.genuine_signatures = self._load_signatures_by_user(self.genuine_dir)
        
        # Get all forged signatures organized by user
        self.forged_signatures = self._load_signatures_by_user(self.forged_dir)
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        print(f"✅ Dataset created:")
        print(f"   - Users: {len(self.genuine_signatures)}")
        print(f"   - Total pairs: {len(self.pairs)}")
        print(f"   - Positive pairs (genuine-genuine): {sum(1 for _, _, label in self.pairs if label == 1)}")
        print(f"   - Negative pairs (genuine-forged): {sum(1 for _, _, label in self.pairs if label == 0)}")
    
    def _load_signatures_by_user(self, directory: Path) -> dict:
        """
        Load all signatures and organize them by user ID
        
        Returns:
            dict: {user_id: [list of image paths]}
        """
        signatures_by_user = {}
        
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            return signatures_by_user
        
        # Get all image files
        image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
        
        for img_path in image_files:
            # Extract user ID from filename
            # Expected format: user_001_genuine_001.png or user_001_forged_001.png
            filename = img_path.stem
            parts = filename.split("_")
            
            if len(parts) >= 2:
                user_id = parts[1]  # Extract user ID (e.g., "001")
                
                if user_id not in signatures_by_user:
                    signatures_by_user[user_id] = []
                
                signatures_by_user[user_id].append(img_path)
        
        return signatures_by_user
    
    def _generate_pairs(self) -> List[Tuple[Path, Path, int]]:
        """
        Generate pairs of signatures for training
        
        Returns:
            List of tuples: (image1_path, image2_path, label)
            label = 1 for genuine pairs, 0 for forged/different user pairs
        """
        pairs = []
        user_ids = list(self.genuine_signatures.keys())
        
        for user_id in user_ids:
            genuine_sigs = self.genuine_signatures.get(user_id, [])
            forged_sigs = self.forged_signatures.get(user_id, [])
            
            # Generate positive pairs (genuine-genuine from same user)
            num_positive = self.pairs_per_user // 2
            for _ in range(num_positive):
                if len(genuine_sigs) >= 2:
                    img1, img2 = random.sample(genuine_sigs, 2)
                    pairs.append((img1, img2, 1))  # Label 1 = same person
            
            # Generate negative pairs (genuine-forged)
            num_negative = self.pairs_per_user // 2
            for _ in range(num_negative):
                if len(genuine_sigs) >= 1:
                    # Try genuine-forged first
                    if len(forged_sigs) >= 1:
                        img1 = random.choice(genuine_sigs)
                        img2 = random.choice(forged_sigs)
                        pairs.append((img1, img2, 0))  # Label 0 = different/forged
                    else:
                        # If no forged signatures, use genuine from different user
                        other_users = [uid for uid in user_ids if uid != user_id]
                        if other_users:
                            other_user = random.choice(other_users)
                            other_genuine = self.genuine_signatures[other_user]
                            if other_genuine:
                                img1 = random.choice(genuine_sigs)
                                img2 = random.choice(other_genuine)
                                pairs.append((img1, img2, 0))
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self) -> int:
        """Return total number of pairs"""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of signatures
        
        Args:
            idx: Index of the pair
            
        Returns:
            img1: First signature image (tensor)
            img2: Second signature image (tensor)
            label: 1 if same person, 0 if different/forged (tensor)
        """
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        
        return img1, img2, label
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        """
        Load and preprocess a signature image
        
        Args:
            img_path: Path to the image
            
        Returns:
            Preprocessed image as tensor
        """
        # Read image
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to fixed size
        img = cv2.resize(img, (220, 155))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add channel dimension
        img = torch.from_numpy(img).unsqueeze(0)  # Shape: (1, H, W)
        
        return img


def get_transforms(augment: bool = False):
    """
    Get image transforms for training/validation
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        Transform pipeline
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ])
    else:
        transform = None
    
    return transform


def create_dataloaders(data_dir: Path, batch_size: int = 32, 
                       train_split: float = 0.7, val_split: float = 0.15,
                       pairs_per_user: int = 50):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        pairs_per_user: Number of pairs per user
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = SignatureDataset(
        data_dir=data_dir,
        transform=None,
        pairs_per_user=pairs_per_user
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for M1 Mac compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(train_dataset)} pairs")
    print(f"   Validation: {len(val_dataset)} pairs")
    print(f"   Test: {len(test_dataset)} pairs")
    
    return train_loader, val_loader, test_loader


# Test the dataset
if __name__ == "__main__":
    from pathlib import Path
    
    # Test with sample data
    data_dir = Path("../data/raw/sample_signatures")
    
    if data_dir.exists():
        print("Testing SignatureDataset...")
        
        # Create dataset
        dataset = SignatureDataset(data_dir, pairs_per_user=20)
        
        # Get a sample
        img1, img2, label = dataset[0]
        print(f"\nSample pair:")
        print(f"   Image 1 shape: {img1.shape}")
        print(f"   Image 2 shape: {img2.shape}")
        print(f"   Label: {label.item()} ({'Same person' if label == 1 else 'Different/Forged'})")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir, batch_size=16, pairs_per_user=20
        )
        
        print("\n✅ Dataset module working correctly!")
    else:
        print(f"❌ Data directory not found: {data_dir}")
