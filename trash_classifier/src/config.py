import torch

# ==== Cấu hình chung ====
DATASET_PATH = "nitinvermafk/trash-dataset"
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224

NUM_WORKERS = 0  # Windows nên để 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_trash_resnet18.pth"
ACCURACY_THRESHOLD = 90.0
TENSORBOARD_LOG_DIR = "runs/trash_classification"
