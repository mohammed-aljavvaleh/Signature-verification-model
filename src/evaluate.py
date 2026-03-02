"""
Evaluation script for Signature Verification Model
Tests the trained model and calculates metrics
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns

from src.config import Config
from src.dataset import create_dataloaders
from src.model import SiameseNetwork


class Evaluator:
    """Evaluator class for testing the trained model"""
    
    def __init__(self, model, test_loader, config):
        """
        Initialize evaluator
        
        Args:
            model: Trained Siamese Network
            test_loader: Test data loader
            config: Configuration object
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        
        # Move model to device
        self.model = self.model.to(config.DEVICE)
        self.model.eval()
        
        print(f"✅ Evaluator initialized")
        print(f"   Device: {config.DEVICE}")
        print(f"   Test samples: {len(test_loader.dataset)}")
    
    def evaluate(self):
        """Evaluate model on test set"""
        all_distances = []
        all_labels = []
        
        print("\n🧪 Evaluating model on test set...")
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(self.test_loader, desc="Testing"):
                # Move to device
                img1 = img1.to(self.config.DEVICE)
                img2 = img2.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                
                # Forward pass
                embedding1, embedding2 = self.model(img1, img2)
                
                # Calculate distances
                distances = F.pairwise_distance(embedding1, embedding2)
                
                # Store results
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        
        return all_distances, all_labels
    
    def calculate_metrics(self, distances, labels, threshold):
        """
        Calculate classification metrics
        
        Args:
            distances: Array of distances
            labels: Array of labels (1=same, 0=different)
            threshold: Decision threshold
            
        Returns:
            Dictionary of metrics
        """
        # Predictions: distance < threshold means same person (1)
        predictions = (distances < threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': threshold
        }
    
    def find_best_threshold(self, distances, labels):
        """
        Find the best threshold for classification
        
        Args:
            distances: Array of distances
            labels: Array of labels
            
        Returns:
            Best threshold value
        """
        # Try different thresholds
        thresholds = np.linspace(0, max(distances), 100)
        best_f1 = 0
        best_threshold = 0
        
        for threshold in thresholds:
            metrics = self.calculate_metrics(distances, labels, threshold)
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
        
        return best_threshold
    
    def plot_distance_distribution(self, distances, labels):
        """Plot distribution of distances for same vs different pairs"""
        same_distances = distances[labels == 1]
        diff_distances = distances[labels == 0]
        
        plt.figure(figsize=(10, 6))
        plt.hist(same_distances, bins=30, alpha=0.6, label='Same Person', color='green')
        plt.hist(diff_distances, bins=30, alpha=0.6, label='Different/Forged', color='red')
        plt.axvline(self.config.VERIFICATION_THRESHOLD, color='black', linestyle='--', 
                   label=f'Threshold ({self.config.VERIFICATION_THRESHOLD})', linewidth=2)
        plt.xlabel('Euclidean Distance', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Distances', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.config.RESULTS_DIR / 'distance_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distance distribution saved: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, distances, labels):
        """Plot ROC curve"""
        # For ROC, we need to flip distances (lower distance = same person)
        # So we use -distances as scores
        fpr, tpr, thresholds = roc_curve(labels, -distances)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.config.RESULTS_DIR / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curve saved: {save_path}")
        plt.close()
        
        return roc_auc
    
    def plot_confusion_matrix(self, distances, labels, threshold):
        """Plot confusion matrix"""
        predictions = (distances < threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Different/Forged', 'Same Person'],
                   yticklabels=['Different/Forged', 'Same Person'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.config.RESULTS_DIR / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved: {save_path}")
        plt.close()
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Evaluate
        distances, labels = self.evaluate()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total pairs: {len(labels)}")
        print(f"   Same person pairs: {sum(labels == 1)}")
        print(f"   Different/Forged pairs: {sum(labels == 0)}")
        
        print(f"\n📏 Distance Statistics:")
        same_distances = distances[labels == 1]
        diff_distances = distances[labels == 0]
        print(f"   Same person - Mean: {np.mean(same_distances):.4f}, Std: {np.std(same_distances):.4f}")
        print(f"   Different - Mean: {np.mean(diff_distances):.4f}, Std: {np.std(diff_distances):.4f}")
        
        # Find best threshold
        print(f"\n🔍 Finding optimal threshold...")
        best_threshold = self.find_best_threshold(distances, labels)
        print(f"   Best threshold: {best_threshold:.4f}")
        
        # Calculate metrics with default threshold
        print(f"\n📈 Metrics (threshold = {self.config.VERIFICATION_THRESHOLD}):")
        metrics_default = self.calculate_metrics(distances, labels, self.config.VERIFICATION_THRESHOLD)
        for key, value in metrics_default.items():
            if key != 'threshold':
                print(f"   {key.capitalize()}: {value:.4f}")
        
        # Calculate metrics with best threshold
        print(f"\n📈 Metrics (optimal threshold = {best_threshold:.4f}):")
        metrics_best = self.calculate_metrics(distances, labels, best_threshold)
        for key, value in metrics_best.items():
            if key != 'threshold':
                print(f"   {key.capitalize()}: {value:.4f}")
        
        # Plot results
        print(f"\n📊 Generating plots...")
        self.plot_distance_distribution(distances, labels)
        roc_auc = self.plot_roc_curve(distances, labels)
        self.plot_confusion_matrix(distances, labels, best_threshold)
        
        print(f"\n🎯 ROC AUC Score: {roc_auc:.4f}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        
        return metrics_best, best_threshold


def main():
    """Main evaluation function"""
    print("="*60)
    print("SIGNATURE VERIFICATION - MODEL EVALUATION")
    print("="*60)
    
    # Load data
    print("\n📂 Loading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=Config.RAW_DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        train_split=Config.TRAIN_SPLIT,
        val_split=Config.VAL_SPLIT,
        pairs_per_user=Config.PAIRS_PER_USER
    )
    
    # Load trained model
    print("\n🧠 Loading trained model...")
    model = SiameseNetwork(embedding_dim=Config.EMBEDDING_DIM)
    
    model_path = Config.MODELS_DIR / Config.BEST_MODEL_NAME
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first using: python3 -m src.train")
        return
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from: {model_path}")
    print(f"   Trained for {checkpoint['epoch']+1} epochs")
    print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, Config)
    
    # Run evaluation
    metrics, best_threshold = evaluator.run_full_evaluation()
    
    print(f"\n💡 Recommendation:")
    print(f"   Use threshold = {best_threshold:.4f} for best performance")
    print(f"   Expected accuracy: {metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
