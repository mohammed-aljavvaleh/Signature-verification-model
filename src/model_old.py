"""
Siamese Neural Network for Signature Verification
Uses twin CNNs to extract features and compare signatures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network for signature verification
    
    Architecture:
    - Twin CNN branches (shared weights)
    - Each branch extracts features from a signature
    - Features are compared using Euclidean distance
    - Contrastive loss trains the network
    """
    
    def __init__(self, embedding_dim=128):
        """
        Initialize the Siamese Network
        
        Args:
            embedding_dim: Dimension of the output feature vector
        """
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # CNN Feature Extractor (shared between both branches)
        self.feature_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 155x220 -> 77x110
            nn.Dropout2d(0.2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 77x110 -> 38x55
            nn.Dropout2d(0.2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 38x55 -> 19x27
            nn.Dropout2d(0.3),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 19x27 -> 9x13
            nn.Dropout2d(0.3),
        )
        
        # Calculate flattened size
        # After 4 max pools: 155/16 ≈ 9, 220/16 ≈ 13
        # So output is approximately 9x13x256
        self.flattened_size = 9 * 13 * 256
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, embedding_dim),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x):
        """
        Forward pass for one signature image
        
        Args:
            x: Input image tensor (batch_size, 1, H, W)
            
        Returns:
            Embedding vector (batch_size, embedding_dim)
        """
        # Extract features with CNN
        x = self.feature_extractor(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get embedding
        x = self.fc(x)
        
        return x
    
    def forward(self, img1, img2):
        """
        Forward pass for a pair of signature images
        
        Args:
            img1: First signature (batch_size, 1, H, W)
            img2: Second signature (batch_size, 1, H, W)
            
        Returns:
            embedding1: Feature vector for img1
            embedding2: Feature vector for img2
        """
        # Get embeddings for both images
        embedding1 = self.forward_one(img1)
        embedding2 = self.forward_one(img2)
        
        return embedding1, embedding2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function
    
    Used to train Siamese networks. Pulls similar pairs closer and
    pushes dissimilar pairs apart with a margin.
    
    Formula:
    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    
    Where:
    - Y = 0 for similar pairs (genuine-genuine)
    - Y = 1 for dissimilar pairs (genuine-forged)
    - D = Euclidean distance between embeddings
    """
    
    def __init__(self, margin=1.0):
        """
        Initialize Contrastive Loss
        
        Args:
            margin: Distance margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Calculate contrastive loss
        
        Args:
            embedding1: Feature vector from first image
            embedding2: Feature vector from second image
            label: 1 if same person (similar), 0 if different (dissimilar)
            
        Returns:
            Contrastive loss value
        """
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        # For label=1 (same person): minimize distance
        # For label=0 (different person): maximize distance up to margin
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("="*60)
    print("SIAMESE NETWORK MODEL TEST")
    print("="*60)
    
    # Create model
    model = SiameseNetwork(embedding_dim=128)
    
    print(f"\n✅ Model created successfully!")
    print(f"📊 Total trainable parameters: {count_parameters(model):,}")
    
    # Test with dummy input
    batch_size = 4
    img1 = torch.randn(batch_size, 1, 155, 220)
    img2 = torch.randn(batch_size, 1, 155, 220)
    
    print(f"\n🧪 Testing forward pass...")
    print(f"   Input 1 shape: {img1.shape}")
    print(f"   Input 2 shape: {img2.shape}")
    
    # Forward pass
    embedding1, embedding2 = model(img1, img2)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Embedding 1 shape: {embedding1.shape}")
    print(f"   Embedding 2 shape: {embedding2.shape}")
    
    # Test loss function
    criterion = ContrastiveLoss(margin=1.0)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)  # Mixed labels
    
    loss = criterion(embedding1, embedding2, labels)
    print(f"\n✅ Loss calculation successful!")
    print(f"   Loss value: {loss.item():.4f}")
    
    # Calculate distances
    distances = F.pairwise_distance(embedding1, embedding2)
    print(f"\n📏 Euclidean distances:")
    for i, (dist, label) in enumerate(zip(distances, labels)):
        pair_type = "Same person" if label == 1 else "Different person"
        print(f"   Pair {i+1}: {dist.item():.4f} ({pair_type})")
    
    print("\n" + "="*60)
    print("✅ MODEL TEST COMPLETE!")
    print("="*60)
