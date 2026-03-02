"""
Fixed Dataset loader for CEDAR Signature Dataset
Handles the specific CEDAR filename format: XXX_YY-ZZZ_WW.jpg
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
import re


class CEDARDataset(Dataset):
    """
    Dataset class for CEDAR signature dataset
    
    CEDAR filename format: 095_38-004_09.jpg
    - 095: forger/sample ID
    - 38: writer ID (actual person)
    - 004: signature number
    - 09: instance/variation
    """
    
    def __init__(self, data_dir: Path, transform=None, pairs_per_user: int = 30):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs_per_user = pairs_per_user
        
        self.genuine_dir = data_dir / "genuine"
        self.forged_dir = data_dir / "forged"
        
        # Load signatures organized by writer ID
        self.genuine_signatures = self._load_signatures_by_writer(self.genuine_dir)
        self.forged_signatures = self._load_signatures_by_writer(self.forged_dir)
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        print(f"✅ CEDAR Dataset loaded:")
        print(f"   - Writers: {len(self.genuine_signatures)}")
        print(f"   - Total pairs: {len(self.pairs)}")
        print(f"   - Positive (same): {sum(1 for _, _, l in self.pairs if l == 1)}")
        print(f"   - Negative (diff): {sum(1 for _, _, l in self.pairs if l == 0)}")
    
    def _extract_writer_id(self, filename: str) -> str:
        """
        Extract writer ID from CEDAR filename
        
        Format: XXX_YY-ZZZ_WW.jpg
        We want: YY (the writer ID)
        """
        # Remove extension
        name = filename.replace('.jpg', '').replace('.png', '')
        
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 2:
            # The writer ID is the second part (after first underscore)
            writer_id = parts[1].split('-')[0]  # Take before the dash
            return writer_id
        
        return "unknown"
    
    def _load_signatures_by_writer(self, directory: Path) -> dict:
        """Load signatures organized by writer ID"""
        signatures_by_writer = {}
        
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            return signatures_by_writer
        
        # Get all image files
        image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
        
        for img_path in image_files:
            writer_id = self._extract_writer_id(img_path.name)
            
            if writer_id not in signatures_by_writer:
                signatures_by_writer[writer_id] = []
            
            signatures_by_writer[writer_id].append(img_path)
        
        # Remove writers with too few signatures
        signatures_by_writer = {
            k: v for k, v in signatures_by_writer.items() 
            if len(v) >= 2  # Need at least 2 signatures per writer
        }
        
        return signatures_by_writer
    
    def _generate_pairs(self) -> List[Tuple[Path, Path, int]]:
        """Generate pairs of signatures"""
        pairs = []
        writer_ids = list(self.genuine_signatures.keys())
        
        print(f"\n📊 Generating pairs for {len(writer_ids)} writers...")
        
        for writer_id in writer_ids:
            genuine_sigs = self.genuine_signatures.get(writer_id, [])
            
            if len(genuine_sigs) < 2:
                continue
            
            # Generate POSITIVE pairs (genuine-genuine from SAME writer)
            num_positive = self.pairs_per_user // 2
            for _ in range(num_positive):
                if len(genuine_sigs) >= 2:
                    img1, img2 = random.sample(genuine_sigs, 2)
                    pairs.append((img1, img2, 1))  # Label 1 = same writer
            
            # Generate NEGATIVE pairs (genuine vs forged OR different writer)
            num_negative = self.pairs_per_user // 2
            for _ in range(num_negative):
                if len(genuine_sigs) >= 1:
                    img1 = random.choice(genuine_sigs)
                    
                    # Try to get forged signature from same writer
                    forged_sigs = self.forged_signatures.get(writer_id, [])
                    
                    if forged_sigs and random.random() < 0.5:
                        # Use forged signature
                        img2 = random.choice(forged_sigs)
                    else:
                        # Use genuine from different writer
                        other_writers = [w for w in writer_ids if w != writer_id]
                        if other_writers:
                            other_writer = random.choice(other_writers)
                            other_sigs = self.genuine_signatures[other_writer]
                            if other_sigs:
                                img2 = random.choice(other_sigs)
                            else:
                                continue
                        else:
                            continue
                    
                    pairs.append((img1, img2, 0))  # Label 0 = different
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return img1, img2, label
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise ValueError(f"Could not load: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (220, 155))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img


def create_dataloaders(data_dir: Path, batch_size: int = 64, 
                       train_split: float = 0.7, val_split: float = 0.15,
                       pairs_per_user: int = 30):
    """Create dataloaders for CEDAR dataset"""
    
    full_dataset = CEDARDataset(
        data_dir=data_dir,
        transform=None,
        pairs_per_user=pairs_per_user
    )
    
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(train_dataset)} pairs")
    print(f"   Val: {len(val_dataset)} pairs")
    print(f"   Test: {len(test_dataset)} pairs")
    
    return train_loader, val_loader, test_loader


# Test
if __name__ == "__main__":
    from pathlib import Path
    
    data_dir = Path("../data/raw/cedar")
    
    if data_dir.exists():
        print("Testing CEDAR Dataset Loader...")
        
        dataset = CEDARDataset(data_dir, pairs_per_user=20)
        
        # Check a sample
        img1, img2, label = dataset[0]
        print(f"\nSample:")
        print(f"   Image 1: {img1.shape}")
        print(f"   Image 2: {img2.shape}")
        print(f"   Label: {label.item()}")
        
        # Check some actual pairs
        print(f"\nFirst 10 pairs:")
        for i in range(min(10, len(dataset.pairs))):
            p1, p2, lbl = dataset.pairs[i]
            w1 = dataset._extract_writer_id(p1.name)
            w2 = dataset._extract_writer_id(p2.name)
            match = "✅" if (w1 == w2 and lbl == 1) or (w1 != w2 and lbl == 0) else "❌"
            print(f"   {match} {p1.name[:20]:20} vs {p2.name[:20]:20} | writers: {w1} vs {w2} | label: {lbl}")
        
        print("\n✅ Dataset loader working!")
