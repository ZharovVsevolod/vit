from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from . import config
import lightning as L
import cv2
import torch
import einops
from datasets import load_dataset, load_from_disk
import albumentations as A
import numpy as np

class Dataset_for_Images(Dataset):
    def __init__(self, image_paths, labels, need_rearrange=True, need_albumantations = config.NEED_ALBUMENTATION) -> None:
        super().__init__()

        self.image_paths = image_paths
        self.labels = labels
        self.need_rearrange = need_rearrange
        self.need_albumantations = need_albumantations
        if self.need_albumantations:
            self.aug = A.Compose({
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=(-90, 90)),
                A.VerticalFlip(p=0.5)
            })
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = image.astype("float32") / 255.0
        if self.need_albumantations:
            image = np.array(image)
            image = self.aug(image=image)["image"]

        image = torch.tensor(image)
        if self.need_rearrange:
            image = einops.rearrange(image, "h w c -> c w h", c=3)

        return image, self.labels[index]

class ViT_DataModule(L.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def load_data_from_hf(self):
        ds = load_dataset(path=config.DATASET_NAME, split=config.DATASET_SPLIT[0], cache_dir=config.CASHE_DIRECTORY)
        ds.save_to_disk(config.IMAGES_TRAIN_PATH)

        ds = load_dataset(path=config.DATASET_NAME, split=config.DATASET_SPLIT[1], cache_dir=config.CASHE_DIRECTORY)
        ds.save_to_disk(config.IMAGES_VAL_PATH)

        ds = load_dataset(path=config.DATASET_NAME, split=config.DATASET_SPLIT[2], cache_dir=config.CASHE_DIRECTORY)
        ds.save_to_disk(config.IMAGES_TEST_PATH)
    
    def prepare_data(self) -> None:
        self.load_data_from_hf()
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            ds_train = load_from_disk(config.IMAGES_TRAIN_PATH)
            self.train_ds = Dataset_for_Images(ds_train["image_file_path"], ds_train["labels"])

            ds_val = load_from_disk(config.IMAGES_VAL_PATH)
            self.val_ds = Dataset_for_Images(ds_val["image_file_path"], ds_val["labels"], need_albumantations=False)

        if stage == "test" or stage is None:
            ds_test = load_from_disk(config.IMAGES_TEST_PATH)
            self.test_ds = Dataset_for_Images(ds_test["image_file_path"], ds_test["labels"], need_albumantations=False)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=config.BATCH_SIZE
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=config.BATCH_SIZE
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=config.BATCH_SIZE
        )