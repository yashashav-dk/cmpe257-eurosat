"""Single source of truth for paths, seeds, and hyperparameters."""
import os
import torch

# ─── Reproducibility ───
SEED = 42
SEEDS = [42, 123, 456]

# ─── Paths (resolved at import time) ───
PROJECT_ROOT = os.environ.get("EUROSAT_PROJECT_ROOT",
                               os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroSAT_RGB")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# ─── Dataset ───
NUM_CLASSES = 10
IMG_SIZE = 64
RESNET_IMG_SIZE = 224
NUM_CHANNELS = 3
TOTAL_IMAGES = 27000
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

# ─── Splits ───
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Normalization ───
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
EUROSAT_MEAN = [0.3448, 0.3806, 0.4083]   # auto-computed by EDA
EUROSAT_STD  = [0.2031, 0.1371, 0.1156]   # auto-computed by EDA

# ─── Training ───
BATCH_SIZE = 64
NUM_EPOCHS_CNN = 100
NUM_EPOCHS_RESNET = 50
EARLY_STOP_PATIENCE = 10
CHECKPOINT_EVERY = 5

# ─── Classical ML ───
PCA_COMPONENTS = [50, 100, 200, 500]
SVM_C_VALUES = [0.1, 1, 10, 100]
SVM_GAMMA_VALUES = [1e-3, 1e-2, 1e-1, "scale"]
CV_FOLDS = 5

# ─── Exp B (optimizer ablation) ───
OPTIMIZER_LR_GRID = [1e-4, 1e-3, 1e-2]

# ─── Exp C (regularization ablation) ───
L2_LAMBDA_VALUES = [1e-4, 1e-3]
DROPOUT_P_VALUES = [0.3, 0.5]
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1,
    },
}

# ─── Device ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
