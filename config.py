"""
Configuration file for pneumonia detection pipeline.
Modify these settings to customize the training process.
"""

# ============================================
# DATASET CONFIGURATION
# ============================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "chest_xray"

MODEL_DIR = BASE_DIR / "saved_models"

OUTPUT_DIR = BASE_DIR / "outputs"

ASSETS_DIR = BASE_DIR / "assets"

# ============================================
# IMAGE PREPROCESSING
# ============================================
IMAGE_SIZE = 224              # Target image size (224x224 - standard CNN input)
                             # Can reduce to 128 for faster training

# ============================================
# TRAINING CONFIGURATION
# ============================================
BATCH_SIZE = 16              # Batch size - increase for faster training, decrease for less memory
LEARNING_RATE = 0.001        # Adam optimizer learning rate
                             # Decrease to 0.0005 if training is unstable
MAX_EPOCHS = 100             # Maximum epochs - training stops earlier with early stopping
EARLY_STOPPING_PATIENCE = 10 # Stop if validation loss doesn't improve for N epochs

# ============================================
# MODEL ARCHITECTURE
# ============================================
DROPOUT_CONV = 0.25          # Dropout for convolutional layers
DROPOUT_DENSE = 0.5          # Dropout for dense layers
NUM_CONV_FILTERS = [32, 64, 128, 128]  # Filters per conv block
DENSE_UNITS = [256, 128]     # Units in dense layers

# ============================================
# OUTPUT CONFIGURATION
# ============================================
MODEL_SAVE_PATH = 'pneumonia_model.h5'
ACCURACY_GRAPH_PATH = 'accuracy_loss_graph.png'
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'

# ============================================
# VISUALIZATION
# ============================================
PLOT_DPI = 300               # Resolution for saved plots
PLOT_FIGSIZE_METRICS = (14, 5)   # Figure size for accuracy/loss plots
PLOT_FIGSIZE_CM = (8, 6)     # Figure size for confusion matrix

# ============================================
# CLASS NAMES
# ============================================
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
