"""
Configuration file for Signature Verification AI Project
Contains all hyperparameters and settings
"""
import torch
from pathlib import Path

class Config:
    """Configuration class for the project"""
    
    # ============================================================
    # PATHS
    # ============================================================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw" / "cedar"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Create directories if they don't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # DATA SETTINGS
    # ============================================================
    # Image dimensions
    IMG_HEIGHT = 155
    IMG_WIDTH = 220
    IMG_CHANNELS = 1  # Grayscale
    
    # Dataset split ratios
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Number of signature pairs to generate
    PAIRS_PER_USER = 50  # Total pairs generated per user
    
    # ============================================================
    # MODEL HYPERPARAMETERS
    # ============================================================
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 50
    
    # Early stopping
    PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    # Contrastive Loss margin
    MARGIN = 1.0  # Distance margin for dissimilar pairs
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    # CNN parameters
    EMBEDDING_DIM = 128  # Size of the feature vector output
    
    # Convolutional layers configuration
    CONV_FILTERS = [32, 64, 128, 256]  # Number of filters in each conv layer
    KERNEL_SIZE = 3
    POOL_SIZE = 2
    DROPOUT_RATE = 0.3
    
    # ============================================================
    # DEVICE CONFIGURATION
    # ============================================================
    # Use MPS (Metal Performance Shaders) for M1 Mac if available
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("✅ Using MPS (GPU acceleration on M1 Mac)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("✅ Using CUDA (NVIDIA GPU)")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️  Using CPU (slower training)")
    
    # ============================================================
    # EVALUATION SETTINGS
    # ============================================================
    # Threshold for signature verification
    # If distance < threshold -> Same person (genuine)
    # If distance > threshold -> Different person (forged)
    VERIFICATION_THRESHOLD = 0.5
    
    # ============================================================
    # LOGGING & CHECKPOINTS
    # ============================================================
    # Save model every N epochs
    CHECKPOINT_FREQUENCY = 5
    
    # Model save name
    MODEL_NAME = "siamese_signature_model.pth"
    BEST_MODEL_NAME = "best_siamese_model.pth"
    
    # TensorBoard logging
    USE_TENSORBOARD = True
    LOG_DIR = RESULTS_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # DATA AUGMENTATION
    # ============================================================
    USE_AUGMENTATION = True
    
    # Augmentation parameters
    ROTATION_RANGE = 10  # degrees
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    
    # ============================================================
    # RANDOM SEED (for reproducibility)
    # ============================================================
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("\n" + "="*60)
        print("SIGNATURE VERIFICATION - CONFIGURATION")
        print("="*60)
        
        print("\n📁 PATHS:")
        print(f"   Data Directory: {cls.RAW_DATA_DIR}")
        print(f"   Models Directory: {cls.MODELS_DIR}")
        print(f"   Results Directory: {cls.RESULTS_DIR}")
        
        print("\n🖼️  IMAGE SETTINGS:")
        print(f"   Image Size: {cls.IMG_HEIGHT}x{cls.IMG_WIDTH}")
        print(f"   Channels: {cls.IMG_CHANNELS} (Grayscale)")
        
        print("\n🧠 MODEL SETTINGS:")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Learning Rate: {cls.LEARNING_RATE}")
        print(f"   Epochs: {cls.NUM_EPOCHS}")
        print(f"   Embedding Dimension: {cls.EMBEDDING_DIM}")
        print(f"   Conv Filters: {cls.CONV_FILTERS}")
        
        print("\n💻 DEVICE:")
        print(f"   Using: {cls.DEVICE}")
        
        print("\n✅ VERIFICATION:")
        print(f"   Threshold: {cls.VERIFICATION_THRESHOLD}")
        
        print("\n" + "="*60 + "\n")
    
    @classmethod
    def set_seed(cls):
        """Set random seed for reproducibility"""
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(cls.RANDOM_SEED)


# Test the configuration
if __name__ == "__main__":
    Config.print_config()
    Config.set_seed()
    print("✅ Configuration loaded successfully!")
