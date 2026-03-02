"""
Training script for Signature Verification Siamese Network
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.dataset import create_dataloaders
from src.model import SiameseNetwork, ContrastiveLoss


class Trainer:
    """Trainer class for Siamese Network"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        """
        Initialize trainer
        
        Args:
            model: Siamese Network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        
        # Training tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # TensorBoard
        if config.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        else:
            self.writer = None
        
        # Move model to device
        self.model = self.model.to(config.DEVICE)
        
        print(f"\n✅ Trainer initialized")
        print(f"   Device: {config.DEVICE}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
            # Move to device
            img1 = img1.to(self.config.DEVICE)
            img2 = img2.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embedding1, embedding2 = self.model(img1, img2)
            
            # Calculate loss
            loss = self.criterion(embedding1, embedding2, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Training/Batch_Loss', loss.item(), step)
        
        # Average loss for epoch
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                img1 = img1.to(self.config.DEVICE)
                img2 = img2.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                
                # Forward pass
                embedding1, embedding2 = self.model(img1, img2)
                
                # Calculate loss
                loss = self.criterion(embedding1, embedding2, labels)
                val_loss += loss.item()
        
        # Average loss
        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        save_path = self.config.MODELS_DIR / filename
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved: {save_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            print(f"\n📊 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Print results
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.config.BEST_MODEL_NAME)
                print(f"   ✅ New best model! (Val Loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {self.config.PATIENCE} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.CHECKPOINT_FREQUENCY == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
        print(f"✅ Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(self.config.MODEL_NAME)
        
        # Plot training curves
        self.plot_training_curves()
        
        if self.writer:
            self.writer.close()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = self.config.RESULTS_DIR / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Training curves saved: {save_path}")
        plt.close()


def main():
    """Main training function"""
    # Print configuration
    Config.print_config()
    
    # Set random seed
    Config.set_seed()
    
    print("\n" + "="*60)
    print("INITIALIZING TRAINING")
    print("="*60)
    
    # Create dataloaders
    print("\n📂 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Config.RAW_DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        train_split=Config.TRAIN_SPLIT,
        val_split=Config.VAL_SPLIT,
        pairs_per_user=Config.PAIRS_PER_USER
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = SiameseNetwork(embedding_dim=Config.EMBEDDING_DIM)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    
    # Loss function
    criterion = ContrastiveLoss(margin=Config.MARGIN)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=Config
    )
    
    # Start training
    trainer.train()
    
    print("\n✅ Training script completed successfully!")


if __name__ == "__main__":
    main()
