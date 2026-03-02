"""
Improved Configuration for CEDAR Dataset
Better hyperparameters to prevent overfitting
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
    RAW_DATA_DIR = DATA_DIR / "raw" / "cedar"  # Changed to CEDAR
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # DATA SETTINGS
    # ============================================================
    IMG_HEIGHT = 155
    IMG_WIDTH = 220
    IMG_CHANNELS = 1
    
    # Dataset split ratios
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Reduced pairs to prevent overfitting
    PAIRS_PER_USER = 30  # Reduced from 50
    
    # ============================================================
    # MODEL HYPERPARAMETERS - IMPROVED
    # ============================================================
    BATCH_SIZE = 32  # Increased for better generalization
    LEARNING_RATE = 0.0001  # Reduced learning rate
    NUM_EPOCHS = 100
    
    # Early stopping
    PATIENCE = 15  # Increased patience
    
    # Contrastive Loss margin
    MARGIN = 2.0  # Increased margin
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    EMBEDDING_DIM = 128
    
    # Simplified architecture
    CONV_FILTERS = [32, 64, 128, 128]  # Reduced complexity
    KERNEL_SIZE = 3
    POOL_SIZE = 2
    DROPOUT_RATE = 0.5  # Increased dropout
    
    # ============================================================
    # DEVICE CONFIGURATION
    # ============================================================
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
    VERIFICATION_THRESHOLD = 0.5
    
    # ============================================================
    # LOGGING & CHECKPOINTS
    # ============================================================
    CHECKPOINT_FREQUENCY = 10
    
    MODEL_NAME = "siamese_cedar_model.pth"
    BEST_MODEL_NAME = "best_siamese_cedar_model.pth"
    
    USE_TENSORBOARD = True
    LOG_DIR = RESULTS_DIR / "logs_cedar"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # DATA AUGMENTATION
    # ============================================================
    USE_AUGMENTATION = False  # Disable for now to reduce noise
    
    ROTATION_RANGE = 5
    WIDTH_SHIFT_RANGE = 0.05
    HEIGHT_SHIFT_RANGE = 0.05
    ZOOM_RANGE = 0.05
    
    # ============================================================
    # RANDOM SEED
    # ============================================================
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        print("\n" + "="*60)
        print("SIGNATURE VERIFICATION - IMPROVED CONFIG (CEDAR)")
        print("="*60)
        
        print("\n📁 PATHS:")
        print(f"   Data: {cls.RAW_DATA_DIR}")
        print(f"   Models: {cls.MODELS_DIR}")
        
        print("\n🧠 MODEL SETTINGS:")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Learning Rate: {cls.LEARNING_RATE}")
        print(f"   Epochs: {cls.NUM_EPOCHS}")
        print(f"   Patience: {cls.PATIENCE}")
        print(f"   Margin: {cls.MARGIN}")
        
        print("\n💻 DEVICE:")
        print(f"   {cls.DEVICE}")
        
        print("\n" + "="*60 + "\n")
    
    @classmethod
    def set_seed(cls):
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(cls.RANDOM_SEED)


if __name__ == "__main__":
    Config.print_config()
    Config.set_seed()
    print("✅ Configuration loaded!")
