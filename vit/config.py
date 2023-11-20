import torch
import os

# Конфигурация ViT
IMAGE_SIZE:int = 224
PATCH_SIZE:int = 16
IN_CHANNELS:int = 3
NUM_CLASSES:int = 1000
EMBEDDING_DIM:int = 768
DEPTH:int = 12
NUM_HEADS:int = 12
MLP_RATIO:float = 4.0
QKV_BIAS:bool = False
DROP_RATE:float = 0.0
ONLY_CLASS_PREDICTION:bool = True

# Определение device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Некоторые гиперпараметры
LEARNIG_RATE:float = 1e-5
NUM_EPOCHS = 100
BATCH_SIZE = 128

# Пути входа и выхода данных
BASE_OUTPUT = "output"
IMAGES_TRAIN_PATH = os.path.join("dataset", "train")
IMAGES_VAL_PATH = os.path.join("dataset", "validation")
IMAGES_TEST_PATH = os.path.join("dataset", "test")
MODEL_PATH = os.path.join(BASE_OUTPUT, "saved_model")
TRAINER_ROOT_DIR = os.path.join(BASE_OUTPUT, "checkpoints")