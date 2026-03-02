"""
Improved Siamese Neural Network with better regularization
Prevents overfitting on CEDAR dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Improved Siamese Neural Network with regularization
    """
    
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Simpler CNN with more regularization
        self.feature_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),  # Increased dropout
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),  # Increased dropout
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),  # Increased dropout
            
            # Conv Block 4 - Smaller filters
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Reduced from 256 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5),  # Strong dropout
        )
        
        # Calculate flattened size
        self.flattened_size = 9 * 13 * 128  # Updated for 128 filters
        
        # Simpler FC layers with strong regularization
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),  # Reduced from 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Very strong dropout
            
            nn.Linear(256, embedding_dim),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward(self, img1, img2):
        embedding1 = self.forward_one(img1)
        embedding2 = self.forward_one(img2)
        return embedding1, embedding2


class ContrastiveLoss(nn.Module):
    """Contrastive Loss Function"""
    
    def __init__(self, margin=2.0):  # Increased margin
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*60)
    print("IMPROVED SIAMESE NETWORK MODEL TEST")
    print("="*60)
    
    model = SiameseNetwork(embedding_dim=128)
    
    print(f"\n✅ Model created!")
    print(f"📊 Total parameters: {count_parameters(model):,}")
    
    batch_size = 4
    img1 = torch.randn(batch_size, 1, 155, 220)
    img2 = torch.randn(batch_size, 1, 155, 220)
    
    embedding1, embedding2 = model(img1, img2)
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Embedding shape: {embedding1.shape}")
    
    criterion = ContrastiveLoss(margin=2.0)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    loss = criterion(embedding1, embedding2, labels)
    
    print(f"\n✅ Loss: {loss.item():.4f}")
    print("\n" + "="*60)
