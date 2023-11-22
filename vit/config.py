import torch
import os
from typing import Literal

# Конфигурация ViT
IMAGE_SIZE:int = 500
PATCH_SIZE:int = 20
IN_CHANNELS:int = 3
NUM_CLASSES:int = 3
EMBEDDING_DIM:int = 768
DEPTH:int = 4
NUM_HEADS:int = 8
MLP_RATIO:float = 4.0
QKV_BIAS:bool = False
DROP_RATE:float = 0.1
ONLY_CLASS_PREDICTION:bool = True

# Определение device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CASHE_DIRECTORY = os.path.join("cache", "hf")

# Некоторые гиперпараметры
NUM_EPOCHS:int = 100
BATCH_SIZE:int = 16
NEED_ALBUMENTATION = True
LEARNIG_RATE:float = 1e-5
TYPE_OF_SCHEDULER: Literal["ReduceOnPlateau", "OneCycleLR"] = "OneCycleLR"
LEARNIG_RATE_COEF_FOR_CYCLE:int = 2

# Пути входа и выхода данных
BASE_OUTPUT = "output"
IMAGES_TRAIN_PATH = os.path.join("dataset", "train")
IMAGES_VAL_PATH = os.path.join("dataset", "validation")
IMAGES_TEST_PATH = os.path.join("dataset", "test")
MODEL_PATH = os.path.join(BASE_OUTPUT, "saved_model")
TRAINER_ROOT_DIR = os.path.join(BASE_OUTPUT, "checkpoints")

DATASET_NAME = "beans"
DATASET_SPLIT = ["train", "validation", "test"]